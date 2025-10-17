## src/service/service.py

import inspect
import json
import logging
import os
import time
import warnings
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langsmith import Client as LangsmithClient

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info
from auth.middleware import AuthMiddleware
from core import LoggingMiddleware, settings
from core.langfuse import get_langfuse_client, get_langfuse_handler
from langgraph.types import Command, Interrupt
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
from service import catalog_postgres
from service.files_router import router as files_router
from service.storage import ensure_upload_root
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


@dataclass
class LangfuseTelemetry:
    """Per-request Langfuse telemetry context."""

    trace_id: str
    session_id: str | None
    user_id: str | None
    metadata: dict[str, Any]
    trace: Any | None = None
    handlers: list[Any] = field(default_factory=list)

    def context_metadata(self, **extra: Any) -> dict[str, Any]:
        merged = dict(self.metadata)
        merged.update({k: v for k, v in extra.items() if v is not None})
        return merged

    def callback_handlers(self) -> list[Any]:
        return list(self.handlers)


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


def _safe_get_trace_handler(trace: Any) -> Any | None:
    get_handler = getattr(trace, "get_langchain_handler", None)
    if not callable(get_handler):
        return None
    try:
        return get_handler()
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        logger.debug("Langfuse langchain handler unavailable: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Failed to acquire Langfuse trace handler: %s", exc)
    return None


def _configure_langfuse_handler(handler: Any, telemetry: LangfuseTelemetry) -> None:
    setters = {
        "set_trace_id": telemetry.trace_id,
        "set_session_id": telemetry.session_id,
        "set_user_id": telemetry.user_id,
    }
    for method_name, value in setters.items():
        if value is None:
            continue
        setter = getattr(handler, method_name, None)
        if callable(setter):
            try:
                setter(value)
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Langfuse handler %s failed: %s", method_name, exc)
        attr = method_name.replace("set_", "")
        if hasattr(handler, attr):
            try:
                setattr(handler, attr, value)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Setting handler attribute %s failed: %s", attr, exc)


def _prepare_langfuse_context(
    agent_id: str,
    run_id: UUID,
    thread_id: str,
    user_id: str,
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    metadata_overrides: Mapping[str, Any] | None = None,
) -> LangfuseTelemetry:
    metadata: dict[str, Any] = {"agent_id": agent_id, "thread_id": thread_id, "user_id": user_id}
    if metadata_overrides:
        metadata.update({k: v for k, v in metadata_overrides.items() if v is not None})

    telemetry = LangfuseTelemetry(
        trace_id=str(trace_id or run_id),
        session_id=(session_id or thread_id) or None,
        user_id=user_id,
        metadata=metadata,
    )

    if not settings.LANGFUSE_TRACING:
        return telemetry

    client = get_langfuse_client()
    if client is None:
        return telemetry

    try:
        trace_kwargs = {
            "name": f"agent.{agent_id}.trace",
            "trace_id": telemetry.trace_id,
            "user_id": user_id,
            "metadata": metadata,
        }
        if telemetry.session_id:
            trace_kwargs["session_id"] = telemetry.session_id
        telemetry.trace = client.trace(**trace_kwargs)
    except Exception as exc:  # pragma: no cover - network interaction
        logger.debug("Failed to create Langfuse trace: %s", exc)
    else:
        handler = _safe_get_trace_handler(telemetry.trace)
        if handler is not None:
            _configure_langfuse_handler(handler, telemetry)
            telemetry.handlers.append(handler)

    if not telemetry.handlers:
        handler = get_langfuse_handler()
        if handler is not None:
            _configure_langfuse_handler(handler, telemetry)
            telemetry.handlers.append(handler)

    return telemetry


def _start_langfuse_span(
    name: str,
    telemetry: LangfuseTelemetry,
    *,
    parent: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any | None, Any]:
    span_parent = parent if parent is not None else telemetry.trace or get_langfuse_client()
    if span_parent is None or not hasattr(span_parent, "span"):
        return None, nullcontext()

    span_kwargs: dict[str, Any] = {"name": name}
    if metadata:
        span_kwargs["metadata"] = metadata
    if telemetry.trace is None and parent is None:
        span_kwargs.setdefault("trace_id", telemetry.trace_id)
        if telemetry.session_id:
            span_kwargs.setdefault("session_id", telemetry.session_id)
        if telemetry.user_id:
            span_kwargs.setdefault("user_id", telemetry.user_id)

    try:
        span = span_parent.span(**span_kwargs)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Failed to start Langfuse span '%s': %s", name, exc)
        return None, nullcontext()

    start_cm = getattr(span, "start_as_current_span", None)
    if callable(start_cm):
        try:
            context = start_cm()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Langfuse start_as_current_span failed: %s", exc)
            context = nullcontext()
    else:
        context = nullcontext()

    return span, context


def _update_span(span: Any | None, **payload: Any) -> None:
    if span is None or not payload:
        return
    try:
        span.update(**payload)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Failed to update Langfuse span: %s", exc)


def _format_exception(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _end_span(span: Any | None, error: Exception | None = None) -> None:
    if span is None:
        return
    try:
        if error is None:
            span.end()
        else:
            span.end(level="ERROR", status_message=_format_exception(error))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Failed to end Langfuse span: %s", exc)


def _truncate(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _summarize_chat_message(message: ChatMessage) -> dict[str, Any]:
    data = message.model_dump()
    data["content"] = _truncate(message.content)
    return data


def _serialize_user_input(user_input: UserInput | StreamInput) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": user_input.message,
        "model": str(user_input.model) if user_input.model else None,
        "thread_id": user_input.thread_id,
    }
    if isinstance(user_input, StreamInput):
        payload["stream_tokens"] = user_input.stream_tokens
    if user_input.agent_config:
        payload["agent_config_keys"] = sorted(user_input.agent_config.keys())
    return {k: v for k, v in payload.items() if v is not None}


def _langfuse_stream_payload(telemetry: LangfuseTelemetry, run_id: UUID | str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trace_id": telemetry.trace_id,
        "run_id": str(run_id),
    }
    if telemetry.session_id:
        payload["session_id"] = telemetry.session_id
    if telemetry.user_id:
        payload["user_id"] = telemetry.user_id
    return payload


def _attach_langfuse_metadata(
    message: ChatMessage, telemetry: LangfuseTelemetry, run_id: UUID | str
) -> None:
    if message.run_id is None:
        message.run_id = str(run_id)
    langfuse_meta = _langfuse_stream_payload(telemetry, run_id)
    existing = (
        message.custom_data.get("langfuse")
        if isinstance(message.custom_data, dict)
        else None
    )
    if isinstance(existing, dict):
        existing.update(langfuse_meta)
        langfuse_meta = existing
    message.custom_data["langfuse"] = langfuse_meta


def _record_stream_message_span(
    telemetry: LangfuseTelemetry, parent_span: Any | None, message: ChatMessage
) -> None:
    span, context = _start_langfuse_span(
        "agent.stream.message",
        telemetry,
        parent=parent_span,
        metadata=telemetry.context_metadata(
            event="message", message_type=message.type
        ),
    )
    with context:
        if span is not None:
            _update_span(span, output=_summarize_chat_message(message))
    _end_span(span)


def _record_stream_token_span(
    telemetry: LangfuseTelemetry, parent_span: Any | None, text: str
) -> None:
    span, context = _start_langfuse_span(
        "agent.stream.token",
        telemetry,
        parent=parent_span,
        metadata=telemetry.context_metadata(event="token"),
    )
    with context:
        if span is not None:
            _update_span(span, output={"token": _truncate(text, 200)})
    _end_span(span)


_LANGFUSE_HEALTH_TTL_SECONDS = 30.0
_langfuse_health_cache: dict[str, float | str] | None = None


def _get_cached_langfuse_health() -> str:
    global _langfuse_health_cache
    now = time.monotonic()
    if _langfuse_health_cache and now < float(_langfuse_health_cache.get("expires", 0)):
        return str(_langfuse_health_cache.get("status", "unknown"))

    client = get_langfuse_client()
    if client is None:
        status = "disabled"
    else:
        try:
            status = "connected" if client.auth_check() else "disconnected"
        except Exception as exc:  # pragma: no cover - network interaction
            logger.error(f"Langfuse connection error: {exc}")
            status = "disconnected"

    _langfuse_health_cache = {
        "expires": now + _LANGFUSE_HEALTH_TTL_SECONDS,
        "status": status,
    }
    return status

async def _handle_input(
    user_input: UserInput, agent: AgentGraph, claims: dict, agent_id: str
) -> tuple[dict[str, Any], UUID, LangfuseTelemetry]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.

    NOTE: user identity is derived from JWT claims (request.state.user), not the payload.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())

    # Canonical identity from JWT (no tenant prefix)
    internal_user = _get_user_id_from_claims(claims or {})

    agent_config = dict(user_input.agent_config or {})
    langfuse_overrides = agent_config.pop("langfuse", None)
    if langfuse_overrides is not None and not isinstance(langfuse_overrides, Mapping):
        raise HTTPException(
            status_code=422,
            detail="agent_config.langfuse must be a mapping when provided",
        )

    trace_override = None
    session_override = None
    extra_metadata: dict[str, Any] | None = None
    if isinstance(langfuse_overrides, Mapping):
        trace_value = langfuse_overrides.get("trace_id")
        if trace_value:
            trace_override = str(trace_value)
        session_value = langfuse_overrides.get("session_id")
        if session_value:
            session_override = str(session_value)
        extra_items = {
            key: value
            for key, value in langfuse_overrides.items()
            if key not in {"trace_id", "session_id"}
        }
        if extra_items:
            extra_metadata = dict(extra_items)

    telemetry = _prepare_langfuse_context(
        agent_id,
        run_id,
        thread_id,
        internal_user,
        trace_id=trace_override,
        session_id=session_override,
        metadata_overrides=extra_metadata,
    )

    configurable = {
        "thread_id": thread_id,
        "model": user_input.model,
        "user_id": internal_user,
        "trace_id": telemetry.trace_id,
        "agent_name": agent_id,
        "run_id": str(run_id),
    }
    if telemetry.session_id:
        configurable["session_id"] = telemetry.session_id

    configurable["langfuse"] = {
        key: value
        for key, value in {
            "trace_id": telemetry.trace_id,
            "session_id": telemetry.session_id,
            "agent_name": agent_id,
            "run_id": str(run_id),
            "thread_id": thread_id,
            "user_id": internal_user,
        }.items()
        if value is not None
    }

    callbacks = telemetry.callback_handlers()

    if agent_config:
        if overlap := configurable.keys() & agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(agent_config)

    if extra_metadata:
        configurable["langfuse"].update(
            {k: v for k, v in extra_metadata.items() if v is not None}
        )

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

    return kwargs, run_id, telemetry


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, request: Request, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    Identity is derived from JWT claims (request.state.user). Any client-sent user_id is ignored.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id, telemetry = await _handle_input(
        user_input, agent, getattr(request.state, "user", {}) or {}, agent_id
    )

    span, span_context = _start_langfuse_span(
        "agent.invoke",
        telemetry,
        metadata=telemetry.context_metadata(run_mode="invoke"),
    )

    try:
        with span_context:
            if span is not None:
                _update_span(span, input=_serialize_user_input(user_input))
            response_events: list[tuple[str, Any]] = await agent.ainvoke(
                **kwargs, stream_mode=["updates", "values"]
            )  # type: ignore # fmt: skip
            response_type, response = response_events[-1]
            if response_type == "values":
                output = langchain_to_chat_message(response["messages"][-1])
            elif response_type == "updates" and "__interrupt__" in response:
                output = langchain_to_chat_message(
                    AIMessage(content=response["__interrupt__"][0].value)
                )
            else:
                raise ValueError(f"Unexpected response type: {response_type}")

            if span is not None:
                _update_span(
                    span,
                    metadata=telemetry.context_metadata(run_status="completed"),
                    output={
                        "response_type": response_type,
                        "message": _summarize_chat_message(output),
                    },
                )
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        if span is not None:
            _update_span(span, metadata=telemetry.context_metadata(run_status="error"))
        _end_span(span, error=e)
        raise HTTPException(status_code=500, detail="Unexpected error")
    else:
        _end_span(span)
        output.run_id = str(run_id)
        _attach_langfuse_metadata(output, telemetry, run_id)
        return output


async def message_generator(
    user_input: StreamInput, agent_id: str, claims: dict
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    Identity is derived from JWT claims (request.state.user). Any client-sent user_id is ignored.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id, telemetry = await _handle_input(user_input, agent, claims, agent_id)

    stream_span, stream_context = _start_langfuse_span(
        "agent.stream",
        telemetry,
        metadata=telemetry.context_metadata(run_mode="stream"),
    )

    token_count = 0
    message_count = 0
    error: Exception | None = None

    try:
        with stream_context:
            if stream_span is not None:
                _update_span(stream_span, input=_serialize_user_input(user_input))
            async for stream_event in agent.astream(
                **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
            ):
                if not isinstance(stream_event, tuple):
                    continue
                if len(stream_event) == 3:
                    _, stream_mode, event = stream_event
                else:
                    stream_mode, event = stream_event

                new_messages = []
                if stream_mode == "updates":
                    for node, updates in event.items():
                        if node == "__interrupt__":
                            interrupt: Interrupt
                            for interrupt in updates:
                                new_messages.append(AIMessage(content=interrupt.value))
                            continue
                        updates = updates or {}
                        update_messages = updates.get("messages", [])
                        if node == "supervisor":
                            if isinstance(update_messages[-1], ToolMessage):
                                update_messages = [update_messages[-1]]
                            else:
                                update_messages = []

                        if node in ("research_expert", "math_expert", "team_supervisor"):
                            update_messages = []
                        new_messages.extend(update_messages)

                if stream_mode == "custom":
                    new_messages = [event]

                processed_messages = []
                current_message: dict[str, Any] = {}
                for message in new_messages:
                    if isinstance(message, tuple):
                        key, value = message
                        current_message[key] = value
                    else:
                        if current_message:
                            processed_messages.append(_create_ai_message(current_message))
                            current_message = {}
                        processed_messages.append(message)

                if current_message:
                    processed_messages.append(_create_ai_message(current_message))

                for message in processed_messages:
                    try:
                        chat_message = langchain_to_chat_message(message)
                        chat_message.run_id = str(run_id)
                        _attach_langfuse_metadata(chat_message, telemetry, run_id)
                    except Exception as e:
                        logger.error(f"Error parsing message: {e}")
                        error_payload = {
                            "type": "error",
                            "content": "Unexpected error",
                            "langfuse": _langfuse_stream_payload(telemetry, run_id),
                        }
                        yield f"data: {json.dumps(error_payload)}\n\n"
                        continue
                    if (
                        isinstance(user_input, StreamInput)
                        and chat_message.type == "human"
                        and chat_message.content == user_input.message
                    ):
                        continue
                    if stream_span is not None:
                        _record_stream_message_span(telemetry, stream_span, chat_message)
                    message_count += 1
                    message_payload = {
                        "type": "message",
                        "content": chat_message.model_dump(),
                        "langfuse": _langfuse_stream_payload(telemetry, run_id),
                    }
                    yield f"data: {json.dumps(message_payload)}\n\n"

                if stream_mode == "messages":
                    if not user_input.stream_tokens:
                        continue
                    msg, metadata = event
                    if "skip_stream" in metadata.get("tags", []):
                        continue
                    if not isinstance(msg, AIMessageChunk):
                        continue
                    content = remove_tool_calls(msg.content)
                    if content:
                        text = convert_message_content_to_string(content)

                        if text.startswith("PLOTLY_JSON:"):
                            payload = text[len("PLOTLY_JSON:"):]
                            event_payload = {
                                "type": "event",
                                "event": "plotly",
                                "content": payload,
                                "langfuse": _langfuse_stream_payload(telemetry, run_id),
                            }
                            yield f"data: {json.dumps(event_payload)}\n\n"
                            continue
                        if text.startswith("DATA_URI:"):
                            payload = text[len("DATA_URI:"):]
                            event_payload = {
                                "type": "event",
                                "event": "image",
                                "content": payload,
                                "langfuse": _langfuse_stream_payload(telemetry, run_id),
                            }
                            yield f"data: {json.dumps(event_payload)}\n\n"
                            continue

                        if stream_span is not None:
                            _record_stream_token_span(telemetry, stream_span, text)
                        token_count += 1
                        token_payload = {
                            "type": "token",
                            "content": text,
                            "langfuse": _langfuse_stream_payload(telemetry, run_id),
                        }
                        yield f"data: {json.dumps(token_payload)}\n\n"
    except Exception as e:
        error = e
        logger.error(f"Error in message generator: {e}")
        error_payload = {
            "type": "error",
            "content": "Internal server error",
            "langfuse": _langfuse_stream_payload(telemetry, run_id),
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
    finally:
        if stream_span is not None:
            _update_span(
                stream_span,
                metadata=telemetry.context_metadata(
                    run_status="error" if error else "completed",
                    message_count=message_count,
                    token_count=token_count,
                ),
            )
        _end_span(stream_span, error=error)
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

    langfuse_trace_id = feedback.trace_id or kwargs.get("metadata", {}).get("langfuse_trace_id")
    langfuse_run_id = (
        feedback.langfuse_run_id
        or kwargs.get("metadata", {}).get("langfuse_run_id")
        or feedback.run_id
    )

    client_langfuse = get_langfuse_client()
    if client_langfuse is not None:
        comment = kwargs.get("comment")
        metadata = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else None
        score_payload: dict[str, Any] = {
            "name": feedback.key,
            "value": feedback.score,
            "data_type": "NUMERIC",
        }
        if comment:
            score_payload["comment"] = comment
        if metadata:
            score_payload["metadata"] = metadata
        try:
            create_score = getattr(client_langfuse, "create_score", None)
            score_current_trace = getattr(client_langfuse, "score_current_trace", None)
            if langfuse_trace_id and callable(create_score):
                create_score(trace_id=langfuse_trace_id, **score_payload)
            elif langfuse_run_id and callable(create_score):
                create_score(run_id=langfuse_run_id, **score_payload)
            elif callable(score_current_trace):
                score_current_trace(**score_payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Langfuse feedback scoring failed: %s", exc)

    return FeedbackResponse(
        langfuse_trace_id=langfuse_trace_id,
        langfuse_run_id=langfuse_run_id,
    )


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
        health_status["langfuse"] = _get_cached_langfuse_health()

    return health_status


app.include_router(router)
app.include_router(files_router, prefix="/files")
