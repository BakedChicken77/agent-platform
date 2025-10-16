## src/service/service.py

import os
import inspect
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, FastAPI, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info
from core import settings, get_settings, Settings, LoggingMiddleware
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from service.tracing import complete_run_trace, start_run_trace

from service.auth import verify_jwt
from auth.middleware import AuthMiddleware

from service.files_router import router as files_router
from service.storage import ensure_upload_root
from service import catalog_postgres

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

# CORS configuration
origins = []
if os.getenv("STREAMLIT_APP_URL"):
    origins.append(os.getenv("STREAMLIT_APP_URL"))

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer and store
    based on settings.
    """
    try:
        ensure_upload_root()
        catalog_postgres.init()   # <-- ensure catalog table/indexes exist
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with initialize_database() as saver, initialize_store() as store:
            # Set up both components
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            # Configure agents with both memory components
            agents = get_all_agent_info()
            for a in agents:
                agent = get_agent(a.key)
                # Set checkpointer for thread-scoped memory (conversation history)
                agent.checkpointer = saver
                # Set store for long-term memory (cross-conversation knowledge)
                agent.store = store
            yield
    except Exception as e:
        logger.error(f"Error during database/store initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],#["Authorization", "authorization", "Content-Type"],
)
settings: Settings = get_settings()

app.add_middleware(AuthMiddleware, settings=settings)
app.add_middleware(LoggingMiddleware)

router = APIRouter()

@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )

@router.get("/me")
async def get_me(request: Request):
    return request.state.user


# --- Helper: extract stable per-user ID from JWT claims (aligns with files_router) ---
def _get_user_id_from_claims(claims: dict) -> str:
    """
    Derive a canonical user_id from JWT claims. We avoid tenant scoping here and
    rely on (user_id, thread_id) for partitioning/isolation.
    """
    return str(
        claims.get("oid")
        or claims.get("sub")
        or claims.get("preferred_username")
        or "unknown"
    )
# -------------------------------------------------------------------------------


async def _handle_input(
    user_input: UserInput,
    agent: AgentGraph,
    claims: dict,
    agent_id: str,
) -> tuple[dict[str, Any], UUID, str]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.

    NOTE: user identity is derived from JWT claims (request.state.user), not the payload.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    trace_id = user_input.trace_id or str(run_id)

    # Canonical identity from JWT (no tenant prefix)
    internal_user = _get_user_id_from_claims(claims or {})

    configurable = {
        "thread_id": thread_id,
        "model": user_input.model,
        "user_id": internal_user,
    }
    configurable["trace_id"] = trace_id

    callbacks = []
    if settings.LANGFUSE_TRACING:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()
        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    if settings.LANGFUSE_TRACING:
        start_run_trace(
            run_id=str(run_id),
            trace_id=trace_id,
            agent_id=agent_id,
            user_id=internal_user,
            thread_id=thread_id,
            input_message=user_input.message,
        )

    return kwargs, run_id, trace_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, request: Request, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    Identity is derived from JWT claims (request.state.user). Any client-sent user_id is ignored.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id, trace_id = await _handle_input(
        user_input,
        agent,
        getattr(request.state, "user", {}) or {},
        agent_id,
    )

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(
            **kwargs, stream_mode=["updates", "values"]
        )  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        output.trace_id = trace_id
        if settings.LANGFUSE_TRACING:
            complete_run_trace(
                run_id=str(run_id),
                output_message=output.content,
                error=None,
            )
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        if settings.LANGFUSE_TRACING:
            complete_run_trace(
                run_id=str(run_id),
                output_message=None,
                error=str(e),
            )
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str, claims: dict
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    Identity is derived from JWT claims (request.state.user). Any client-sent user_id is ignored.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id, trace_id = await _handle_input(user_input, agent, claims, agent_id)

    final_output: str | None = None
    error_message: str | None = None
    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # Handle different stream event structures based on subgraphs
            if len(stream_event) == 3:
                # With subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # Without subgraphs: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if node == "supervisor":
                        # Get only the last ToolMessage since it is added by the
                        # langgraph lib and not actual AI output so it won't be an
                        # independent event
                        if isinstance(update_messages[-1], ToolMessage):
                            update_messages = [update_messages[-1]]
                        else:
                            update_messages = []

                    if node in ("research_expert", "math_expert", "team_supervisor"):
                        update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                    chat_message.trace_id = trace_id
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message; drop it
                if isinstance(user_input, StreamInput) and chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                if chat_message.type == "ai" and chat_message.content:
                    final_output = chat_message.content
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # Drop non-LLM nodes
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    text = convert_message_content_to_string(content)

                    # Detect plot payload markers and emit as custom SSE
                    if text.startswith("PLOTLY_JSON:"):
                        payload = text[len("PLOTLY_JSON:"):]
                        yield f"data: {json.dumps({'type':'event', 'event':'plotly', 'content': payload})}\n\n"
                        continue
                    if text.startswith("DATA_URI:"):
                        payload = text[len("DATA_URI:"):]
                        yield f"data: {json.dumps({'type':'event', 'event':'image', 'content': payload})}\n\n"
                        continue

                    # default token streaming
                    yield f"data: {json.dumps({'type':'token', 'content': text})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        error_message = str(e)
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        if settings.LANGFUSE_TRACING:
            complete_run_trace(
                run_id=str(run_id),
                output_message=final_output,
                error=error_message,
            )
        yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, request: Request, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    Identity is derived from JWT claims (request.state.user). Any client-sent user_id is ignored.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id, getattr(request.state, "user", {}) or {}),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    agent: AgentGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


app.include_router(router)
app.include_router(files_router, prefix="/files")
