```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\run_agent.py

import asyncio
from typing import cast
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

from agents import DEFAULT_AGENT, get_agent  # noqa: E402

# The default agent uses StateGraph.compile() which returns CompiledStateGraph
agent = cast(CompiledStateGraph, get_agent(DEFAULT_AGENT))
agent1 = cast(CompiledStateGraph, get_agent('chatbot'))
agent2 = cast(CompiledStateGraph, get_agent('research-assistant'))
agent3 = cast(CompiledStateGraph, get_agent('rag-assistant'))
agent4 = cast(CompiledStateGraph, get_agent('command-agent'))
agent5 = cast(CompiledStateGraph, get_agent('bg-task-agent'))
agent6 = cast(CompiledStateGraph, get_agent('langgraph-supervisor-agent'))
agent7 = cast(CompiledStateGraph, get_agent('knowledge-base-agent'))
agent8 = cast(CompiledStateGraph, get_agent('agent-team'))
agent9 = cast(CompiledStateGraph, get_agent('agent-team2'))

async def main() -> None:
    

    agent1.get_graph(xray=True).draw_png("chatbot.png")
    agent2.get_graph(xray=True).draw_png("research-assistant.png")
    agent3.get_graph(xray=True).draw_png("rag-assistant.png")
    agent4.get_graph(xray=True).draw_png("command-agent.png")
    agent5.get_graph(xray=True).draw_png("bg-task-agent.png")
    agent6.get_graph(xray=True).draw_png("langgraph-supervisor-agent.png")
    agent7.get_graph(xray=True).draw_png("knowledge-base-agent.png")
    agent8.get_graph(xray=True).draw_png("agent-team.png")
    agent9.get_graph(xray=True).draw_png("agent-team2.png")





    # inputs: MessagesState = {
    #     "messages": [HumanMessage("Find me a recipe for chocolate chip cookies")]
    # }
    # result = await agent.ainvoke(
    #     input=inputs,
    #     config=RunnableConfig(configurable={"thread_id": uuid4()}),
    # )
    # result["messages"][-1].pretty_print()

    # # Draw the agent graph as png
    # # requires:
    # # brew install graphviz
    # # export CFLAGS="-I $(brew --prefix graphviz)/include"
    # # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    # # pip install pygraphviz
    # #
    # agent.get_graph().draw_png("agent_diagram.png")


if __name__ == "__main__":
    asyncio.run(main())

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\run_client.py

import asyncio

from client import AgentClient
from core import settings
from schema import ChatMessage


async def amain() -> None:
    #### ASYNC ####
    client = AgentClient(settings.BASE_URL,agent='rag-assistant')

    print("Agent info:")
    print(client.info)

    print("Chat example:")
    response = await client.ainvoke("list all benefits?", model="gpt-4o")
    response.pretty_print()

    print("\nStream example:")
    async for message in client.astream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"ERROR: Unknown type - {type(message)}")


def main() -> None:
    #### SYNC ####
    client = AgentClient(settings.BASE_URL,agent='rag-assistant')

    print("Agent info:")
    print(client.info)

    print("Chat example:")
    response = client.invoke("Tell me about my company", model="gpt-4o")
    response.pretty_print()

    print("\nStream example:")
    for message in client.stream("Share a quick fun fact?"):
        if isinstance(message, str):
            print(message, flush=True, end="")
        elif isinstance(message, ChatMessage):
            print("\n", flush=True)
            message.pretty_print()
        else:
            print(f"ERROR: Unknown type - {type(message)}")


if __name__ == "__main__":
    print("Running in sync mode")
    main()
    print("\n\n\n\n\n")
    print("Running in async mode")
    asyncio.run(amain())

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\run_service.py

import asyncio
import sys

import uvicorn
from dotenv import load_dotenv

from core import settings

load_dotenv()

if __name__ == "__main__":
    # Set Compatible event loop policy on Windows Systems.
    # On Windows systems, the default ProactorEventLoop can cause issues with
    # certain async database drivers like psycopg (PostgreSQL driver).
    # The WindowsSelectorEventLoopPolicy provides better compatibility and prevents
    # "RuntimeError: Event loop is closed" errors when working with database connections.
    # This needs to be set before running the application server.
    # Refer to the documentation for more information.
    # https://www.psycopg.org/psycopg3/docs/advanced/async.html#asynchronous-operations
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("service:app", host=settings.HOST, port=settings.PORT, reload=settings.is_dev())

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\streamlit_app.py

import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData, TaskDataStatus

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            use_streaming = st.toggle("Stream results", value=True)

            # Display user ID (for debugging or user information)
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # Include both thread_id and user_id in the URL for sharing to maintain user identity
            chat_url = (
                f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            )
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.drs.com/AIS-FWB-Engineering/DRS_Agent.git)"
        st.caption(
            "Made with :material/favorite: by [Steve](https://drscom.sharepoint.us/sites/BU03/Dept/Engineering/SitePages/ProjectHome.aspx) for AIS-FWB"
        )

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AIS SEPS.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()  # Clear stale containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                await handle_agent_msgs(messages_agen, call_results, is_new)
                                break
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_agent_msgs(messages_agen, call_results, is_new):
    """
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.
    """
    nested_popovers = {}
    # looking for the Success tool call message
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)
    status = call_results.get(getattr(first_msg, "tool_call_id", None))
    # Process first message
    if status and first_msg.content:
        status.write(first_msg.content)
        # Continue reading until finish_reason='stop'
    while True:
        # Check for completion on current message
        finish_reason = getattr(first_msg, "response_metadata", {}).get("finish_reason")
        # Break out of status container if finish_reason is anything other than "tool_calls"
        if finish_reason is not None and finish_reason != "tool_calls":
            if status:
                status.update(state="complete")
            break
        # Read next message
        sub_msg = await anext(messages_agen)
        # this should only happen is skip_stream flag is removed
        # if isinstance(sub_msg, str):
        #     continue
        if is_new:
            st.session_state.messages.append(sub_msg)

        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.write(sub_msg.content)
            first_msg = sub_msg
            continue
        # Display content and tool calls using the same status
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)
            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    popover = status.popover(f"{tc['name']}", icon="🛠️")
                    popover.write(f"**Tool:** {tc['name']}")
                    popover.write("**Input:**")
                    popover.write(tc["args"])
                    # Store the popover reference using the tool call ID
                    nested_popovers[tc["id"]] = popover
        # Update first_msg for next iteration
        first_msg = sub_msg


if __name__ == "__main__":
    asyncio.run(main())

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\agent_registry.py

from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.command_agent import command_agent
from agents.interrupt_agent import interrupt_agent
from agents.knowledge_base_agent import kb_agent
from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.rag_assistant import rag_assistant
from agents.research_assistant import research_assistant
from agents.agent_team import agent_team 
from agents.agent_team2 import agent_team2

from schema import AgentInfo




DEFAULT_AGENT = "rag-assistant"#"chatbot"
# DEFAULT_AGENT = "research-assistant"

# Type alias to handle LangGraph's different agent patterns
# - @entrypoint functions return Pregel
# - StateGraph().compile() returns CompiledStateGraph
AgentGraph = CompiledStateGraph | Pregel


@dataclass
class Agent:
    description: str
    graph: AgentGraph


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot.", graph=chatbot),
    "research-assistant": Agent(
        description="A research assistant with web search and calculator.", graph=research_assistant
    ),
    "rag-assistant": Agent(
        description="A RAG assistant with access to information in a database.", graph=rag_assistant
    ),
    "command-agent": Agent(description="A command agent.", graph=command_agent),
    "bg-task-agent": Agent(description="A background task agent.", graph=bg_task_agent),
    "langgraph-supervisor-agent": Agent(
        description="A langgraph supervisor agent", graph=langgraph_supervisor_agent
    ),
    "interrupt-agent": Agent(description="An agent the uses interrupts.", graph=interrupt_agent),
    "knowledge-base-agent": Agent(
        description="A retrieval-augmented generation agent using Amazon Bedrock Knowledge Base",
        graph=kb_agent,
    ),
    "agent-team": Agent(                          
        description="Hierarchical research agent team.",
        graph=agent_team,
    ),
    "agent-team2": Agent(                          
        description="Hierarchical research→write agent team.",
        graph=agent_team2,
    ),
}


def get_agent(agent_id: str) -> AgentGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\agent_team.py

"""
Hierarchical Agent Team
=======================

A two‑layer team that first gathers facts with the existing *research_assistant*
sub‑graph and then hands the material to a dedicated writing agent that
crafts the final answer.

The top‑level “supervisor” decides which worker to run next and when the job
is finished, following the pattern shown in the “Hierarchical Agent Teams”
tutorial notebook.
"""

from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from core import get_model, settings
from agents.research_assistant import research_assistant  # existing graph
from langchain_core.messages import AIMessage


# --------------------------------------------------------------------------- #
# ••• State definition                                                        #
# --------------------------------------------------------------------------- #

class TeamState(MessagesState, total=False):
    """Conversation state shared by the team."""
    next: str  # name of the next node selected by the supervisor


# --------------------------------------------------------------------------- #
# ••• Utility: build a supervisor (router) node                               #
# --------------------------------------------------------------------------- #

def _make_supervisor(llm: ChatOpenAI, members: list[str]):
    """
    Build a router that picks the next worker.

    The LLM is asked to answer with one of the *members* names or the token
    `FINISH`.  When it answers `FINISH` the graph transitions to `__end__`.
    """
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor managing the workers: "
        f"{', '.join(members)}.\n"
        "After reading the conversation decide which worker should act next. "
        "Respond **only** with the worker name, or `FINISH` if the task is done."
    )

    class Router(TypedDict):
        next: Literal[*options]

    def _node(state: TeamState) -> Command[Literal[*members, "__end__"]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        choice = llm.with_structured_output(Router).invoke(messages)
        goto = choice["next"]
        if goto == "FINISH":
            goto = END
        return Command(
            goto=goto,
            update={"next": goto, "messages": [AIMessage(content="")]}
        )

    return _node


# --------------------------------------------------------------------------- #
# ••• Worker nodes                                                            #
# --------------------------------------------------------------------------- #

async def _research_worker(state: TeamState) -> Command[Literal["team_supervisor"]]:
    """
    Delegate to the existing `research_assistant` sub‑graph and return its
    final message back to the supervisor.
    """
    response = await research_assistant.ainvoke({"messages": state["messages"]})
    return Command(
        update={"messages": [response["messages"][-1]]},
        goto="team_supervisor",
    )


# A very small “writing” agent: just call the LLM with the conversation so far
_writer_llm = get_model(settings.DEFAULT_MODEL)


def _writer_worker(state: TeamState) -> Command[Literal["team_supervisor"]]:
    """Turn the accumulated research into a clear, concise answer."""
    reply: BaseMessage = _writer_llm.invoke(state["messages"])
    return Command(update={"messages": [reply]}, goto="team_supervisor")


# --------------------------------------------------------------------------- #
# ••• Build and export the compiled graph                                     #
# --------------------------------------------------------------------------- #

def _build_team():
    router_llm = get_model(settings.DEFAULT_MODEL)
    supervisor = _make_supervisor(router_llm, ["research_expert", "writer_worker"])

    builder = StateGraph(TeamState)
    builder.add_node("team_supervisor", supervisor)
    builder.add_node("research_expert", _research_worker)
    builder.add_node("writer_worker", _writer_worker)

    builder.add_edge(START, "team_supervisor")          # entry‑point

    graph = builder.compile()
    graph.name = "hierarchical-agent-team"
    return graph


agent_team = _build_team()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\agent_team2.py

"""
Hierarchical Agent Team
=======================

A three-layer team composed of:
1. **Research subgraph**: search & web-scraping agents.
2. **Writing subgraph**: note-taker, doc-writer, chart-generator.
3. **Top-level supervisor** orchestrating the two subgraphs.

This follows the "Hierarchical Agent Teams" tutorial, using LangGraph and LangChain's create_react_agent.
"""

from typing import Literal, TypedDict, List, Optional, Dict

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.document_loaders import WebBaseLoader
# from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from tempfile import TemporaryDirectory
from pathlib import Path

from core import get_model, settings

from agents.tools_writing import create_outline, read_document, write_document, edit_document, python_repl_tool
from agents.tools import database_search, get_full_doc_text




# ---------------------------------------------------------------------------
# ••• Tools for research subgraph
# ---------------------------------------------------------------------------

# tavily_tool = TavilySearch(max_results=5)

# @tool
# def scrape_webpages(urls: List[str]) -> str:
#     """Scrape provided web pages for detailed information."""
#     loader = WebBaseLoader(urls)
#     docs = loader.load()
#     return "\n\n".join(
#         [
#             f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
#             for doc in docs
#         ]
#     )

# ---------------------------------------------------------------------------
# ••• Tools for document-writing subgraph
# ---------------------------------------------------------------------------

# _TEMP_DIRECTORY = PythonREPL().run  # dummy to silence flake; real TemporaryDirectory import omitted
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)
# Tools create_outline, read_document, write_document, edit_document, python_repl_tool
# would be imported from the existing codebase as needed

# For brevity assume these tools are imported:
# from agents.tools import create_outline, read_document, write_document, edit_document
# and python_repl_tool from utilities

# ---------------------------------------------------------------------------
# ••• State definitions
# ---------------------------------------------------------------------------

class TeamState(MessagesState, total=False):
    """Shared conversation state."""
    next: str

# ---------------------------------------------------------------------------
# ••• Supervisor factory
# ---------------------------------------------------------------------------

def make_supervisor_node(llm: BaseChatModel, members: List[str]):
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor managing the workers: "
        f"{', '.join(members)}. After reading the conversation decide which "
        "worker should act next. Respond ONLY with the worker name, or `FINISH`. "
        "Use python_repl to write and run python code to complete tasks."
    )
    class Router(TypedDict):
        next: Literal[*options]

    def supervisor_node(state: TeamState) -> Command[Literal[*members, "__end__"]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        return Command(goto=goto, update={"next": goto})

    return supervisor_node

# ---------------------------------------------------------------------------
# ••• Research subgraph
# ---------------------------------------------------------------------------

llm = get_model(settings.DEFAULT_MODEL)
search_agent = create_react_agent(llm, tools=[database_search])
get_full_doc_text_agent = create_react_agent(llm, tools=[get_full_doc_text])

async def semantic_search_node(state: TeamState) -> Command[Literal["research_supervisor"]]:
    result = await search_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="research_supervisor",
    )

async def get_full_doc_text_node(state: TeamState) -> Command[Literal["research_supervisor"]]:
    result = await get_full_doc_text_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="research_supervisor",
    )

research_supervisor = make_supervisor_node(llm, ["semantic_search", "get_full_doc_text"])

research_builder = StateGraph(TeamState)
research_builder.add_node("research_supervisor", research_supervisor)
research_builder.add_node("semantic_search", semantic_search_node)
research_builder.add_node("get_full_doc_text", get_full_doc_text_node)
research_builder.add_edge(START, "research_supervisor")
research_graph = research_builder.compile()

# ---------------------------------------------------------------------------
# ••• Document-writing subgraph
# ---------------------------------------------------------------------------

# Assume create_outline, read_document, write_document, edit_document, python_repl_tool imported

doc_writer_agent = create_react_agent(
    llm,
    tools=[write_document, edit_document, read_document],
    prompt=(
        "You can read, write and edit documents based on note-taker's outlines. "
        "Don't ask follow-up questions."
    ),
    name="doc_writer_agent"
)
note_taker_agent = create_react_agent(
    llm,
    tools=[create_outline, read_document],
    prompt=(
        "You can read documents and create outlines for the doc writer. "
        "Don't ask follow-up questions."
    ),
    name="note_taker_agent"
)
chart_agent = create_react_agent(
    llm, 
    tools=[read_document, python_repl_tool],
    name="chart_agent"
)

repl_agent = create_react_agent(
    llm,
    tools=[python_repl_tool],
    name="python_repl_agent",
    prompt="You are a Python REPL. Execute the user’s code and return only the output."
)

async def note_node(state: TeamState) -> Command[Literal["doc_supervisor"]]:
    result = await note_taker_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="doc_supervisor",
    )

async def doc_node(state: TeamState) -> Command[Literal["doc_supervisor"]]:
    result = await doc_writer_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="doc_supervisor",
    )

async def chart_node(state: TeamState) -> Command[Literal["doc_supervisor"]]:
    result = await chart_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content,name="chart_generator")]},
        goto="doc_supervisor",
    )

async def repl_node(state: TeamState) -> Command[Literal["top_supervisor"]]:
    result = await repl_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="top_supervisor",
    )

doc_supervisor = make_supervisor_node(llm, ["note_taker", "doc_writer", "chart_generator"])

doc_builder = StateGraph(TeamState)
doc_builder.add_node("doc_supervisor", doc_supervisor)
doc_builder.add_node("note_taker", note_node)
doc_builder.add_node("doc_writer", doc_node)
doc_builder.add_node("chart_generator", chart_node)
doc_builder.add_edge(START, "doc_supervisor")
doc_graph = doc_builder.compile()

# ---------------------------------------------------------------------------
# ••• Top-level composition
# ---------------------------------------------------------------------------

top_supervisor = make_supervisor_node(llm, ["research_team", "writing_team", "python_repl"])

async def call_research_team(state: TeamState) -> Command[Literal["top_supervisor"]]:
    # feed last research output into research_graph
    response = await research_graph.ainvoke({"messages": state["messages"][-1]})
    return Command(
        update={"messages": [HumanMessage(content=response["messages"][-1].content)]},
        goto="top_supervisor",
    )

async def call_writing_team(state: TeamState) -> Command[Literal["top_supervisor"]]:
    response = await doc_graph.ainvoke({"messages": state["messages"][-1]})
    return Command(
        update={"messages": [HumanMessage(content=response["messages"][-1].content)]},
        goto="top_supervisor",
    )

builder = StateGraph(TeamState)
builder.add_node("top_supervisor", top_supervisor)
builder.add_node("research_team", call_research_team)
builder.add_node("writing_team", call_writing_team)
builder.add_node("python_repl", repl_node)
builder.add_edge(START, "top_supervisor")
agent_team2 = builder.compile()
agent_team2.name = "hierarchical-agent-team"


def visualize_graph(graph, filename='workflow.dot'):
    dot = graphviz.Digraph()

    # Add nodes
    for node in graph.nodes:
        dot.node(node, node)

    # Add edges
    for edge in graph.edges:
        dot.edge(edge[0], edge[1])

    # Save to file and render as PNG
    dot.save(filename)
    dot.render(filename, format='png', cleanup=True)

    dot = graphviz.Digraph()

    # # Custom labels for nodes
    # node_labels = {
    #     "document_search": "Document Search",
    #     "generate": "Generate",
    #     "transform_query": "Transform Query",
    #     "web_search": "Web Search",
    #     "finalize_response": "Finalize Response"
    # }

    node_labels = False
    if node_labels:
        # Add nodes with custom labels
        for node in graph.nodes:
            label = node_labels.get(node, node)  # Default to node ID if no custom label
            dot.node(node, label)
    else:
        # Add nodes
        for node in graph.nodes:
            dot.node(node, node)

    # Add edges
    for edge in graph.edges:
        dot.edge(edge[0], edge[1])

    # Handle conditional edges
    for start_node, branches in graph.branches.items():
        for condition_func, branch in branches.items():
            for condition, end_node in branch.ends.items():
                edge_label = f"{condition_func} -> {condition}"
                # Modify the edge label to remove the redundant part
                dot.edge(start_node, end_node, label=condition_func, style="dashed")

    # Save to file and render as PNG
    dot.save(filename)
    dot.render(filename, format='png', cleanup=True)

# Function to visualize without executing the workflow
# async def visualize_only():
#     import graphviz
#     visualize_graph(workflow)

if __name__ == "__main__":
    import graphviz
    visualize_graph(builder)
```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\agent_team3.py


"""
Hierarchical Agent Team
=======================

A two-layer team that first gathers facts with the existing *research_assistant*
sub-graph and then hands the material to a dedicated writing agent that
crafts the final answer.

The top-level “supervisor” decides which worker to run next and when the job
is finished, following the pattern shown in the “Hierarchical Agent Teams”
tutorial notebook.
"""

from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from core import get_model, settings
from agents.research_assistant import research_assistant  # existing graph
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

# --------------------------------------------------------------------------- #
# ••• State definition                                                        #
# --------------------------------------------------------------------------- #

class TeamState(MessagesState, total=False):
    """Conversation state shared by the team."""
    next: str  # name of the next node selected by the supervisor


# --------------------------------------------------------------------------- #
# ••• Utility: build a supervisor (router) node                               #
# --------------------------------------------------------------------------- #
def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: TeamState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


# --------------------------------------------------------------------------- #
# ••• Worker nodes                                                            #
# --------------------------------------------------------------------------- #

async def _research_worker(state: TeamState) -> Command[Literal["team_supervisor"]]:
    """
    Delegate to the existing `research_assistant` sub‑graph and return its
    final message back to the supervisor.
    """
    response = await research_assistant.ainvoke({"messages": state["messages"]})
    return Command(
        update={"messages": [response["messages"][-1]]},
        goto="team_supervisor",
    )


# A very small “writing” agent: just call the LLM with the conversation so far
_writer_llm = get_model(settings.DEFAULT_MODEL)


def _writer_worker(state: TeamState) -> Command[Literal["team_supervisor"]]:
    """Turn the accumulated research into a clear, concise answer."""
    reply: BaseMessage = _writer_llm.invoke(state["messages"])
    return Command(update={"messages": [reply]}, goto="team_supervisor")


# --------------------------------------------------------------------------- #
# ••• Build and export the compiled graph                                     #
# --------------------------------------------------------------------------- #

def _build_team():
    router_llm = get_model(settings.DEFAULT_MODEL)
    supervisor = make_supervisor_node(router_llm, ["research_expert", "writer_worker"])

    builder = StateGraph(TeamState)
    builder.add_node("team_supervisor", supervisor)
    builder.add_node("research_expert", _research_worker)
    builder.add_node("writer_worker", _writer_worker)

    builder.add_edge(START, "team_supervisor")          # entry‑point

    graph = builder.compile()
    graph.name = "hierarchical-agent-team"
    return graph


agent_team = _build_team()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\chatbot.py

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint

from core import get_model, settings
import truststore
truststore.inject_into_ssl()

@entrypoint()
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str, list[BaseMessage]],
    config: RunnableConfig,
):
    messages = inputs["messages"]
    if previous:
        messages = previous["messages"] + messages

    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await model.ainvoke(messages)
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    )

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\command_agent.py

import random
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


# Define the nodes


def node_a(state: AgentState) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["a", "b"])
    goto: Literal["node_b", "node_c"]
    # this is a replacement for a conditional edge function
    if value == "a":
        goto = "node_b"
    else:
        goto = "node_c"

    # note how Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        # this is the state update
        update={"messages": [AIMessage(content=f"Hello {value}")]},
        # this is a replacement for an edge
        goto=goto,
    )


def node_b(state: AgentState):
    print("Called B")
    return {"messages": [AIMessage(content="Hello B")]}


def node_c(state: AgentState):
    print("Called C")
    return {"messages": [AIMessage(content="Hello C")]}


builder = StateGraph(AgentState)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)
# NOTE: there are no edges between nodes A, B and C!

command_agent = builder.compile()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\helper_writing.py

from typing import List, Optional, Literal
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages
from typing_extensions import TypedDict

class State(MessagesState):
    next: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node
```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\interrupt_agent.py

import logging
from datetime import datetime
from typing import Any

from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from core import get_model, settings

# Added logger
logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    birthdate: datetime | None


def wrap_model(
    model: BaseChatModel | Runnable[LanguageModelInput, Any], system_prompt: BaseMessage
) -> RunnableSerializable[AgentState, Any]:
    preprocessor = RunnableLambda(
        lambda state: [system_prompt] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


background_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant that tells users there zodiac sign.
Provide a one sentence summary of the origin of zodiac signs.
Don't tell the user what their sign is, you are just demonstrating your knowledge on the topic.
""")


async def background(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node is to demonstrate doing work before the interrupt"""

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, background_prompt.format())
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


birthdate_extraction_prompt = SystemMessagePromptTemplate.from_template("""
You are an expert at extracting birthdates from conversational text.

Rules for extraction:
- Look for user messages that mention birthdates
- Consider various date formats (MM/DD/YYYY, YYYY-MM-DD, Month Day, Year)
- Validate that the date is reasonable (not in the future)
- If no clear birthdate was provided by the user, return None
""")


class BirthdateExtraction(BaseModel):
    birthdate: str | None = Field(
        description="The extracted birthdate in YYYY-MM-DD format. If no birthdate is found, this should be None."
    )
    reasoning: str = Field(
        description="Explanation of how the birthdate was extracted or why no birthdate was found"
    )


async def determine_birthdate(
    state: AgentState, config: RunnableConfig, store: BaseStore
) -> AgentState:
    """This node examines the conversation history to determine user's birthdate, checking store first."""

    # Attempt to get user_id for unique storage per user
    user_id = config["configurable"].get("user_id")
    logger.info(f"[determine_birthdate] Extracted user_id: {user_id}")
    namespace = None
    key = "birthdate"
    birthdate = None  # Initialize birthdate

    if user_id:
        # Use user_id in the namespace to ensure uniqueness per user
        namespace = (user_id,)

        # Check if we already have the birthdate in the store for this user
        try:
            result = await store.aget(namespace, key=key)
            # Handle cases where store.aget might return Item directly or a list
            user_data = None
            if result:  # Check if anything was returned
                if isinstance(result, list):
                    if result:  # Check if list is not empty
                        user_data = result[0]
                else:  # Assume it's the Item object directly
                    user_data = result

            if user_data and user_data.value.get("birthdate"):
                # Convert ISO format string back to datetime object
                birthdate_str = user_data.value["birthdate"]
                birthdate = datetime.fromisoformat(birthdate_str) if birthdate_str else None
                # We already have the birthdate, return it
                logger.info(
                    f"[determine_birthdate] Found birthdate in store for user {user_id}: {birthdate}"
                )
                return {
                    "birthdate": birthdate,
                    "messages": [],
                }
        except Exception as e:
            # Log the error or handle cases where the store might be unavailable
            logger.error(f"Error reading from store for namespace {namespace}, key {key}: {e}")
            # Proceed with extraction if read fails
            pass
    else:
        # If no user_id, we cannot reliably store/retrieve user-specific data.
        # Consider logging this situation.
        logger.warning(
            "Warning: user_id not found in config. Skipping persistent birthdate storage/retrieval for this run."
        )

    # If birthdate wasn't retrieved from store, proceed with extraction
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m.with_structured_output(BirthdateExtraction), birthdate_extraction_prompt.format()
    ).with_config(tags=["skip_stream"])
    response: BirthdateExtraction = await model_runnable.ainvoke(state, config)

    # If no birthdate found after extraction attempt, interrupt
    if response.birthdate is None:
        birthdate_input = interrupt(f"{response.reasoning}\nPlease tell me your birthdate?")
        # Re-run extraction with the new input
        state["messages"].append(HumanMessage(birthdate_input))
        # Note: Recursive call might need careful handling of depth or state updates
        return await determine_birthdate(state, config, store)

    # Birthdate found - convert string to datetime
    try:
        birthdate = datetime.fromisoformat(response.birthdate)
    except ValueError:
        # If parsing fails, ask for clarification
        birthdate_input = interrupt(
            "I couldn't understand the date format. Please provide your birthdate in YYYY-MM-DD format."
        )
        # Re-run extraction with the new input
        state["messages"].append(HumanMessage(birthdate_input))
        # Note: Recursive call might need careful handling of depth or state updates
        return await determine_birthdate(state, config, store)

    # Store the newly extracted birthdate only if we have a user_id
    if user_id and namespace:
        # Convert datetime to ISO format string for JSON serialization
        birthdate_str = birthdate.isoformat() if birthdate else None
        try:
            await store.aput(namespace, key, {"birthdate": birthdate_str})
        except Exception as e:
            # Log the error or handle cases where the store write might fail
            logger.error(f"Error writing to store for namespace {namespace}, key {key}: {e}")

    # Return the determined birthdate (either from store or extracted)
    logger.info(f"[determine_birthdate] Returning birthdate {birthdate} for user {user_id}")
    return {
        "birthdate": birthdate,
        "messages": [],
    }


response_prompt = SystemMessagePromptTemplate.from_template("""
You are a helpful assistant.

Known information:
- The user's birthdate is {birthdate_str}

User's latest message: "{last_user_message}"

Based on the known information and the user's message, provide a helpful and relevant response.
If the user asked for their birthdate, confirm it.
If the user asked for their zodiac sign, calculate it and tell them.
Otherwise, respond conversationally based on their message.
""")


async def generate_response(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generates the final response based on the user's query and the available birthdate."""
    birthdate = state.get("birthdate")
    if state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        last_user_message = state["messages"][-1].content
    else:
        last_user_message = ""

    if not birthdate:
        # This should ideally not be reached if determine_birthdate worked correctly and possibly interrupted.
        # Handle cases where birthdate might still be missing.
        return {
            "messages": [
                AIMessage(
                    content="I couldn't determine your birthdate. Could you please provide it?"
                )
            ]
        }

    birthdate_str = birthdate.strftime("%B %d, %Y")  # Format for display

    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(
        m, response_prompt.format(birthdate_str=birthdate_str, last_user_message=last_user_message)
    )
    response = await model_runnable.ainvoke(state, config)

    return {"messages": [AIMessage(content=response.content)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("background", background)
agent.add_node("determine_birthdate", determine_birthdate)
agent.add_node("generate_response", generate_response)

agent.set_entry_point("background")
agent.add_edge("background", "determine_birthdate")
agent.add_edge("determine_birthdate", "generate_response")
agent.add_edge("generate_response", END)

interrupt_agent = agent.compile()
interrupt_agent.name = "interrupt-agent"

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\knowledge_base_agent.py

import logging
import os
from typing import Any

from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps

from core import get_model, settings

logger = logging.getLogger(__name__)


# Define the state
class AgentState(MessagesState, total=False):
    """State for Knowledge Base agent."""

    remaining_steps: RemainingSteps
    retrieved_documents: list[dict[str, Any]]
    kb_documents: str


# Create the retriever
def get_kb_retriever():
    """Create and return a Knowledge Base retriever instance."""
    # Get the Knowledge Base ID from environment
    kb_id = os.environ.get("AWS_KB_ID", "")
    if not kb_id:
        raise ValueError("AWS_KB_ID environment variable must be set")

    # Create the retriever with the specified Knowledge Base ID
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 3,
            }
        },
    )
    return retriever


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model with a system prompt for the Knowledge Base agent."""

    def create_system_message(state):
        base_prompt = """You are a helpful assistant that provides accurate information based on retrieved documents.

        You will receive a query along with relevant documents retrieved from a knowledge base. Use these documents to inform your response.

        Follow these guidelines:
        1. Base your answer primarily on the retrieved documents
        2. If the documents contain the answer, provide it clearly and concisely
        3. If the documents are insufficient, state that you don't have enough information
        4. Never make up facts or information not present in the documents
        5. Always cite the source documents when referring to specific information
        6. If the documents contradict each other, acknowledge this and explain the different perspectives

        Format your response in a clear, conversational manner. Use markdown formatting when appropriate.
        """

        # Check if documents were retrieved
        if "kb_documents" in state:
            # Append document information to the system prompt
            document_prompt = f"\n\nI've retrieved the following documents that may be relevant to the query:\n\n{state['kb_documents']}\n\nPlease use these documents to inform your response to the user's query. Only use information from these documents and clearly indicate when you are unsure."
            return [SystemMessage(content=base_prompt + document_prompt)] + state["messages"]
        else:
            # No documents were retrieved
            no_docs_prompt = (
                "\n\nNo relevant documents were found in the knowledge base for this query."
            )
            return [SystemMessage(content=base_prompt + no_docs_prompt)] + state["messages"]

    preprocessor = RunnableLambda(
        create_system_message,
        name="StateModifier",
    )
    return RunnableSequence(preprocessor, model)


async def retrieve_documents(state: AgentState, config: RunnableConfig) -> AgentState:
    """Retrieve relevant documents from the knowledge base."""
    # Get the last human message
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        # Include messages from original state
        return {"messages": [], "retrieved_documents": []}

    # Use the last human message as the query
    query = human_messages[-1].content

    try:
        # Initialize the retriever
        retriever = get_kb_retriever()

        # Retrieve documents
        retrieved_docs = await retriever.ainvoke(query)

        # Create document summaries for the state
        document_summaries = []
        for i, doc in enumerate(retrieved_docs, 1):
            summary = {
                "id": doc.metadata.get("id", f"doc-{i}"),
                "source": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", f"Document {i}"),
                "content": doc.page_content,
                "relevance_score": doc.metadata.get("score", 0),
            }
            document_summaries.append(summary)

        logger.info(f"Retrieved {len(document_summaries)} documents for query: {query[:50]}...")

        return {"retrieved_documents": document_summaries, "messages": []}

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {"retrieved_documents": [], "messages": []}


async def prepare_augmented_prompt(state: AgentState, config: RunnableConfig) -> AgentState:
    """Prepare a prompt augmented with retrieved document content."""
    # Get retrieved documents
    documents = state.get("retrieved_documents", [])

    if not documents:
        return {"messages": []}

    # Format retrieved documents for the model
    formatted_docs = "\n\n".join(
        [
            f"--- Document {i + 1} ---\n"
            f"Source: {doc.get('source', 'Unknown')}\n"
            f"Title: {doc.get('title', 'Unknown')}\n\n"
            f"{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        ]
    )

    # Store formatted documents in the state
    return {"kb_documents": formatted_docs, "messages": []}


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate a response based on the retrieved documents."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)

    response = await model_runnable.ainvoke(state, config)

    return {"messages": [response]}


# Define the graph
agent = StateGraph(AgentState)

# Add nodes
agent.add_node("retrieve_documents", retrieve_documents)
agent.add_node("prepare_augmented_prompt", prepare_augmented_prompt)
agent.add_node("model", acall_model)

# Set entry point
agent.set_entry_point("retrieve_documents")

# Add edges to define the flow
agent.add_edge("retrieve_documents", "prepare_augmented_prompt")
agent.add_edge("prepare_augmented_prompt", "model")
agent.add_edge("model", END)

# Compile the agent
kb_agent = agent.compile()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\langgraph_supervisor_agent.py

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from agents.tools import calculator, database_search, get_full_doc_text

from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )


math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time.",
).with_config(tags=["skip_stream"])

# research_agent = create_react_agent(
#     model=model,
#     tools=[web_search],
#     name="research_expert",
#     prompt="You are a world class researcher with access to web search. Do not do any math.",
# ).with_config(tags=["skip_stream"])


research_agent = create_react_agent(
    model=model,
    tools=[database_search, get_full_doc_text],
    name="research_expert",
    prompt=f"""\
You are a highly capable research assistant. You have access to the following tools:

* **`Database_Search`** — Performs semantic search across Leonardo DRS, Inc.'s official Employee Handbook and Operational Process documents.
* **`Get_Full_Doc_Text`** — Retrieves the complete text of a document by its exact filename (e.g., `'SEP-04-01(M) Process for Product Development.docx'`).

---

### Tool Usage Instructions:

1. **Always begin** with `Database_Search` to identify relevant documents via semantic search.
2. For every document returned, **immediately retrieve its full text** using `Get_Full_Doc_Text`.
3. **Do not rely solely** on the semantic search output — it is incomplete. Your answers **must be based on the full document text**.

NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.
---

### Response Guidelines:

* Cite your sources with **markdown-formatted links** using **only URLs returned by the tools**. Limit to **1-2 citations** per response unless more are essential.
* Be direct, thorough, and precise. **Do not speculate.** Only respond based on verified content from retrieved documents.
""",
).with_config(tags=["skip_stream"])


# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        f"""\
You are a team supervisor managing a research expert and a math expert to support employees by researching \
Leonardo DRS, Inc official Employee Handbook and Operational Process documents.
For Employee Handbook and Operational Process information, use research_agent.
For math problems, use math_agent.\
"""
    ),
    add_handoff_back_messages=False,
)




langgraph_supervisor_agent = workflow.compile()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\llama_guard.py

from enum import Enum

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from core import get_model, settings
from schema.models import GroqModelName


class SafetyAssessment(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"


class LlamaGuardOutput(BaseModel):
    safety_assessment: SafetyAssessment = Field(description="The safety assessment of the content.")
    unsafe_categories: list[str] = Field(
        description="If content is unsafe, the list of unsafe categories.", default=[]
    )


unsafe_content_categories = {
    "S1": "Violent Crimes.",
    "S2": "Non-Violent Crimes.",
    "S3": "Sex Crimes.",
    "S4": "Child Exploitation.",
    "S5": "Defamation.",
    "S6": "Specialized Advice.",
    "S7": "Privacy.",
    "S8": "Intellectual Property.",
    "S9": "Indiscriminate Weapons.",
    "S10": "Hate.",
    "S11": "Self-Harm.",
    "S12": "Sexual Content.",
    "S13": "Elections.",
    "S14": "Code Interpreter Abuse.",
}

categories_str = "\n".join([f"{k}: {v}" for k, v in unsafe_content_categories.items()])

llama_guard_instructions = f"""
Task: Check if there is unsafe content in '{{role}}' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{categories_str}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{{conversation_history}}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST {{role}} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories."""


def parse_llama_guard_output(output: str) -> LlamaGuardOutput:
    if output == "safe":
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
    parsed_output = output.split("\n")
    if len(parsed_output) != 2 or parsed_output[0] != "unsafe":
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.ERROR)
    try:
        categories = parsed_output[1].split(",")
        readable_categories = [unsafe_content_categories[c.strip()].strip(".") for c in categories]
        return LlamaGuardOutput(
            safety_assessment=SafetyAssessment.UNSAFE,
            unsafe_categories=readable_categories,
        )
    except KeyError:
        return LlamaGuardOutput(safety_assessment=SafetyAssessment.ERROR)


class LlamaGuard:
    def __init__(self) -> None:
        if settings.GROQ_API_KEY is None:
            print("GROQ_API_KEY not set, skipping LlamaGuard")
            self.model = None
            return
        self.model = get_model(GroqModelName.LLAMA_GUARD_4_12B).with_config(tags=["skip_stream"])
        self.prompt = PromptTemplate.from_template(llama_guard_instructions)

    def _compile_prompt(self, role: str, messages: list[AnyMessage]) -> str:
        role_mapping = {"ai": "Agent", "human": "User"}
        messages_str = [
            f"{role_mapping[m.type]}: {m.content}" for m in messages if m.type in ["ai", "human"]
        ]
        conversation_history = "\n\n".join(messages_str)
        return self.prompt.format(role=role, conversation_history=conversation_history)

    def invoke(self, role: str, messages: list[AnyMessage]) -> LlamaGuardOutput:
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
        compiled_prompt = self._compile_prompt(role, messages)
        result = self.model.invoke([HumanMessage(content=compiled_prompt)])
        return parse_llama_guard_output(str(result.content))

    async def ainvoke(self, role: str, messages: list[AnyMessage]) -> LlamaGuardOutput:
        if self.model is None:
            return LlamaGuardOutput(safety_assessment=SafetyAssessment.SAFE)
        compiled_prompt = self._compile_prompt(role, messages)
        result = await self.model.ainvoke([HumanMessage(content=compiled_prompt)])
        return parse_llama_guard_output(str(result.content))


if __name__ == "__main__":
    llama_guard = LlamaGuard()
    output = llama_guard.invoke(
        "Agent",
        [
            HumanMessage(content="What's a good way to harm an animal?"),
            AIMessage(
                content="There are many ways to harm animals, but some include hitting them with a stick, throwing rocks at them, or poisoning them."
            ),
        ],
    )
    print(output)

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\rag_assistant.py

from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import database_search, get_full_doc_text
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


tools = [database_search,get_full_doc_text]


current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
You are Leonardo Bot, a helpful and knowledgeable virtual assistant designed to support employees by retrieving \
and answering questions based on Leonardo DRS, Inc official Employee Handbook and Operational documents. Your primary role is to provide \
accurate, concise, and friendly information about company policies, values, procedures, and employee resources. \

You have access to the following tools:
* **`Database_Search`** — Performs semantic search across Leonardo DRS, Inc.'s official Employee Handbook and Operational Process documents.
* **`Get_Full_Doc_Text`** — Retrieves the complete text of a document by its exact filename (e.g., `'SEP-04-01(M) Process for Product Development.docx'`).

---

### Tool Usage Instructions:

1. **Always begin** with `Database_Search` to identify relevant documents via semantic search.
2. For every document returned, **immediately retrieve its full text** using `Get_Full_Doc_Text` so that you understand the full context of the document.
3. **Do not rely solely** on the semantic search output — it is incomplete. Your answers **must be based on the full document text**.

NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

---

After obtaining enough information from the database, generate a comprehensive and informative answer for the \
given question based solely on the search results. You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text.

Multiple searches should be performed if there is nothing in the context relevant to the question at hand.

A few things to remember:
- If multiple sources are required to form a complete and accurate answer, gather information before crafting your response.
- You can search databases multiple times if needed to obtain enough information for a complete answer.
- Include markdown-formatted links to any citations used in your response. Only include one \
or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
- Only use information from the database. Do not use information from outside sources.
"""

# REMEMBER: Cite search results using [${{number}}] notation. If there is no relevant information within the context, just say "Hmm, I'm not seeing it in the SEP documentation." Don't \
# try to make up an answer. Instead, ask follow-up question(s) to better understand what the employee needs. Anything between the preceding 'context' html blocks is retrieved from a \
# knowledge bank, not part of the conversation with the user.\
# """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [format_safety_message(safety_output)],
            "safety": safety_output,
        }

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

rag_assistant = agent.compile()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\research_assistant.py

from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator, database_search, get_full_doc_text


from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


web_search = DuckDuckGoSearchResults(name="WebSearch")
# tools = [web_search, calculator]
# tools = [database_search, calculator, Get_Full_Doc_Text]
tools = [database_search, get_full_doc_text]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""\
You are a highly capable research assistant. You have access to the following tools:

* **`Database_Search`** — Performs semantic search across Leonardo DRS, Inc.'s official Employee Handbook and Operational Process documents.
* **`Get_Full_Doc_Text`** — Retrieves the complete text of a document by its exact filename (e.g., `'SEP-04-01(M) Process for Product Development.docx'`).
* **`Calculator`** — Executes mathematical expressions using `numexpr`.

---

### Tool Usage Instructions:

1. **Always begin** with `Database_Search` to identify relevant documents via semantic search.
2. For every document returned, **immediately retrieve its full text** using `Get_Full_Doc_Text`.
3. **Do not rely solely** on the semantic search output — it is incomplete. Your answers **must be based on the full document text**.
4. **Use `Calculator`** for any mathematical operations required to answer the question.

NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.
---

### Response Guidelines:

* Cite your sources with **markdown-formatted links** using **only URLs returned by the tools**. Limit to **1-2 citations** per response unless more are essential.
* When presenting math results, use **human-readable expressions**, e.g., `"300 * 200 = 60,000"` — not NumExpr format.
* Be direct, thorough, and precise. **Do not speculate.** Only respond based on verified content from retrieved documents.
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})


research_assistant = agent.compile()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\tools.py

import math
import re
import os

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import AzureOpenAIEmbeddings
from core import settings
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import PGVector


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
# def format_contexts(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
def format_contexts(docs):
    segments = []
    count = 1
    for d in docs:
        metadata = getattr(d, 'metadata', {}) or {}
        source = metadata.get('filename', 'unknown')
        url = metadata.get('file_path', 'unknown')
        # url = os.path.basename(source) if source else 'unknown'
        page = metadata.get('page_number', 'N/A')
        content = getattr(d, 'page_content', '[No content]')
        segments.append(f"""
Document: [{url}]({source})  
Page Number: `{page}`  
Page Content:
``` 
{content}
```

---
"""
        )

        count += 1
    return "\n\n".join(segments)



def load_chroma_db():
    # Create the embedding function for our project description database
    try:
        embeddings = AzureOpenAIEmbeddings(
                        api_key=settings.AZURE_OPENAI_API_KEY,
                        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                        model=settings.AZURE_OPENAI_EMBEDDER,
                        api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    except Exception as e:
        raise RuntimeError(
            "Failed to initialize AzureOpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def load_postgre_db():
    # Create the embedding function for our project description database
    try:
        embeddings = AzureOpenAIEmbeddings(
                        api_key=settings.AZURE_OPENAI_API_KEY,
                        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                        model=settings.AZURE_OPENAI_EMBEDDER,
                        api_version=settings.AZURE_OPENAI_API_VERSION,
        )

    except Exception as e:
        raise RuntimeError(
            "Failed to initialize AzureOpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    pg_url = settings.PGVECTOR_URL
    # Load the stored vector database
    postgre_db = PGVector(
        embedding_function=embeddings,
        connection_string=pg_url,
        collection_name="SEPS",)
    retriever = postgre_db.as_retriever(search_kwargs={"k": 5})
    return retriever

# def database_search_func(query: str) -> str:
#     """Searches chroma_db for information in the company's handbook."""
#     # Get the chroma retriever
#     retriever = load_chroma_db()

#     # Search the database for relevant documents
#     documents = retriever.invoke(query)

#     # Format the documents into a string
#     context_str = format_contexts(documents)

#     return context_str

def database_search_func(query: str) -> str:
    """Searches PGVector DB for information in the company's handbook."""

    retriever = load_postgre_db()
    documents = retriever.invoke(query)
    context_str = format_contexts(documents)
    return context_str



def get_full_doc_text_func(file_name: str) -> str:
    """Return the complete text (HTML if available) of a document stored in langchain_pg_embedding.

    Args:
        file_name (str): Exact filename stored in cmetadata->>'filename'.

    Returns:
        str: Concatenated text of all pages or an error string if nothing found.
    """
    try:
        engine = create_engine(settings.PGVECTOR_URL)
    except Exception as e:
        return f"Database connection error: {e}"

    query = text("""
        WITH collection_uuid AS (
            SELECT uuid
            FROM langchain_pg_collection
            WHERE name = 'SEPS'
        )
        SELECT cmetadata->>'page_number' AS page_number,
               document AS doc,
               cmetadata->>'text_as_html' AS html
        FROM langchain_pg_embedding
        WHERE collection_id = (SELECT uuid FROM collection_uuid)
          AND cmetadata->>'filename' = :file_name
        ORDER BY (cmetadata->>'page_number')::int;
    """)

    try:
        with engine.begin() as conn:
            rows = conn.execute(query, {"file_name": file_name}).mappings().all()
    except Exception as e:
        return f"Query execution failed: {e}"

    if not rows:
        return f"No document found for filename '{file_name}'."

    full_text_segments = []
    for row in rows:
        if row["html"]:
            full_text_segments.append(f"```html\n{row['html']}\n```")
        elif row["doc"]:
            full_text_segments.append(row["doc"])

    return "\n\n".join(full_text_segments)






database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database


get_full_doc_text: BaseTool = tool(get_full_doc_text_func)
get_full_doc_text.name = "Get_Full_Doc_Text"
```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\tools_writing.py

from pathlib import Path
# from tempfile import TemporaryDirectory
from typing import Dict, Optional
from typing import Annotated, List


from langchain_core.tools import tool


from langchain_experimental.utilities import PythonREPL
from langchain_experimental.tools.python.tool import PythonREPLTool
import math

# _TEMP_DIRECTORY = TemporaryDirectory()
# WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)
WORKING_DIRECTORY = Path('.\pythonrepl_sandbox')


@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# Warning: This executes code locally, which can be unsafe when not sandboxed


# 1. Build a locked‑down REPL
safe_globals = {
    "__builtins__": {"print": print},
    "math": math,
}
repl = PythonREPL(_globals=safe_globals, _locals={})


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\utils.py

from typing import Any

from langchain_core.messages import ChatMessage
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field


class CustomData(BaseModel):
    "Custom data being sent by an agent"

    data: dict[str, Any] = Field(description="The custom data")

    def to_langchain(self) -> ChatMessage:
        return ChatMessage(content=[self.data], role="custom")

    def dispatch(self, writer: StreamWriter) -> None:
        writer(self.to_langchain())

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\__init__.py

from agents.agent_registry import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info

__all__ = ["get_agent", "get_all_agent_info", "DEFAULT_AGENT", "AgentGraph"]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\bg_task_agent\bg_task_agent.py

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter

from agents.bg_task_agent.task import Task
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: state["messages"],
        name="StateModifier",
    )
    return preprocessor | model  # type: ignore[return-value]


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def bg_task(state: AgentState, writer: StreamWriter) -> AgentState:
    task1 = Task("Simple task 1...", writer)
    task2 = Task("Simple task 2...", writer)

    task1.start()
    await asyncio.sleep(2)
    task2.start()
    await asyncio.sleep(2)
    task1.write_data(data={"status": "Still running..."})
    await asyncio.sleep(2)
    task2.finish(result="error", data={"output": 42})
    await asyncio.sleep(2)
    task1.finish(result="success", data={"output": 42})
    return {"messages": []}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("bg_task", bg_task)
agent.set_entry_point("bg_task")

agent.add_edge("bg_task", "model")
agent.add_edge("model", END)

bg_task_agent = agent.compile()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\agents\bg_task_agent\task.py

from typing import Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.types import StreamWriter

from agents.utils import CustomData
from schema.task_data import TaskData


class Task:
    def __init__(self, task_name: str, writer: StreamWriter | None = None) -> None:
        self.name = task_name
        self.id = str(uuid4())
        self.state: Literal["new", "running", "complete"] = "new"
        self.result: Literal["success", "error"] | None = None
        self.writer = writer

    def _generate_and_dispatch_message(self, writer: StreamWriter | None, data: dict):
        writer = writer or self.writer
        task_data = TaskData(name=self.name, run_id=self.id, state=self.state, data=data)
        if self.result:
            task_data.result = self.result
        task_custom_data = CustomData(
            type=self.name,
            data=task_data.model_dump(),
        )
        if writer:
            task_custom_data.dispatch(writer)
        return task_custom_data.to_langchain()

    def start(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        self.state = "new"
        task_message = self._generate_and_dispatch_message(writer, data)
        return task_message

    def write_data(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        if self.state == "complete":
            raise ValueError("Only incomplete tasks can output data.")
        self.state = "running"
        task_message = self._generate_and_dispatch_message(writer, data)
        return task_message

    def finish(
        self,
        result: Literal["success", "error"],
        writer: StreamWriter | None = None,
        data: dict = {},
    ) -> BaseMessage:
        self.state = "complete"
        self.result = result
        task_message = self._generate_and_dispatch_message(writer, data)
        return task_message

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\client\client.py

import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
)


class AgentClientError(Exception):
    pass


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
            get_info (bool, optional): Whether to fetch agent information on init.
                Default: True
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            AnyMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent synchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent

        Returns:
            ChatMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # Convert the JSON formatted message to an AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # Yield the str token directly
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        Stream the agent's response synchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if user_id:
            request.user_id = user_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        Stream the agent's response asynchronously.

        Each intermediate message of the agent process is yielded as an AnyMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming modelsas they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            user_id (str, optional): User ID for continuing a conversation across multiple threads
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        Create a feedback record for a run.

        This is a simple wrapper for the LangSmith create_feedback API, so the
        credentials can be stored and managed in the service rather than the client.
        See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
        """
        request = ChatHistoryInput(thread_id=thread_id)
        try:
            response = httpx.post(
                f"{self.base_url}/history",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\client\__init__.py

from client.client import AgentClient, AgentClientError

__all__ = ["AgentClient", "AgentClientError"]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\core\llm.py

from functools import cache
from typing import TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from core.settings import settings
from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    VertexAIModelName,
)

_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in AzureOpenAIModelName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in VertexAIModelName}
    | {m: m.value for m in GroqModelName}
    | {m: m.value for m in AWSModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in OpenRouterModelName}
    | {m: m.value for m in FakeModelName}
)


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools):
        return self


ModelT: TypeAlias = (
    AzureChatOpenAI
    | ChatOpenAI
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | ChatVertexAI
    | ChatGroq
    | ChatBedrock
    | ChatOllama
    | FakeToolModel
)


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=settings.TEMPERATURE, streaming=True)
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL,
            temperature=settings.TEMPERATURE,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
        )
    if model_name in AzureOpenAIModelName:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")

        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=api_model_name,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=settings.TEMPERATURE,
            streaming=True,
            timeout=60,
            max_retries=3,
        )
    if model_name in DeepseekModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=settings.TEMPERATURE,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=settings.TEMPERATURE, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=settings.TEMPERATURE, streaming=True)
    if model_name in VertexAIModelName:
        return ChatVertexAI(model=api_model_name, temperature=settings.TEMPERATURE, streaming=True)
    if model_name in GroqModelName:
        if model_name == GroqModelName.LLAMA_GUARD_4_12B:
            return ChatGroq(model=api_model_name, temperature=0.0)
        return ChatGroq(model=api_model_name, temperature=settings.TEMPERATURE)
    if model_name in AWSModelName:
        return ChatBedrock(model_id=api_model_name, temperature=settings.TEMPERATURE)
    if model_name in OllamaModelName:
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(
                model=settings.OLLAMA_MODEL, temperature=settings.TEMPERATURE, base_url=settings.OLLAMA_BASE_URL
            )
        else:
            chat_ollama = ChatOllama(model=settings.OLLAMA_MODEL, temperature=settings.TEMPERATURE)
        return chat_ollama
    if model_name in OpenRouterModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=settings.TEMPERATURE,
            streaming=True,
            base_url="https://openrouter.ai/api/v1/",
            api_key=settings.OPENROUTER_API_KEY,
        )
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\core\settings.py

import os
from enum import StrEnum
from json import loads
from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    Provider,
    VertexAIModelName,
)


class DatabaseType(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGO = "mongo"


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "http://localhost"#"0.0.0.0"
    PORT: int = 8080

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    USE_AWS_BEDROCK: bool = False
    OLLAMA_MODEL: str | None = None
    OLLAMA_BASE_URL: str | None = None
    USE_FAKE_MODEL: bool = False
    OPENROUTER_API_KEY: str | None = None

    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    # Set openai compatible api, mainly used for proof of concept
    COMPATIBLE_MODEL: str | None = None
    COMPATIBLE_API_KEY: SecretStr | None = None
    COMPATIBLE_BASE_URL: str | None = None

    OPENWEATHERMAP_API_KEY: SecretStr | None = None

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    LANGFUSE_TRACING: bool = False
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None

    # Database Configuration
    DATABASE_TYPE: DatabaseType = (
        DatabaseType.SQLITE
    )  # Options: DatabaseType.SQLITE or DatabaseType.POSTGRES
    SQLITE_DB_PATH: str = "checkpoints.db"

    # PostgreSQL Configuration
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_APPLICATION_NAME: str = "agent-service-toolkit"
    POSTGRES_MIN_CONNECTIONS_PER_POOL: int = 1
    POSTGRES_MAX_CONNECTIONS_PER_POOL: int = 1
    PGVECTOR_URL: str | None = None

    # MongoDB Configuration
    MONGO_HOST: str | None = None
    MONGO_PORT: int | None = None
    MONGO_DB: str | None = None
    MONGO_USER: str | None = None
    MONGO_PASSWORD: SecretStr | None = None
    MONGO_AUTH_SOURCE: str | None = None

    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: SecretStr | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_EMBEDDER: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_MAP: dict[str, str] = Field(
        default_factory=dict, description="Map of model names to Azure deployment IDs"
    )
    
    # LLM Settings
    TEMPERATURE: float = 0.05

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.OPENAI_COMPATIBLE: self.COMPATIBLE_BASE_URL and self.COMPATIBLE_MODEL,
            Provider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY,
            Provider.VERTEXAI: self.GOOGLE_APPLICATION_CREDENTIALS,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.AWS: self.USE_AWS_BEDROCK,
            Provider.OLLAMA: self.OLLAMA_MODEL,
            Provider.FAKE: self.USE_FAKE_MODEL,
            Provider.AZURE_OPENAI: self.AZURE_OPENAI_API_KEY,
            Provider.OPENROUTER: self.OPENROUTER_API_KEY,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.OPENAI_COMPATIBLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
                    self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                case Provider.DEEPSEEK:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DeepseekModelName.DEEPSEEK_CHAT
                    self.AVAILABLE_MODELS.update(set(DeepseekModelName))
                case Provider.ANTHROPIC:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AnthropicModelName.HAIKU_3
                    self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GoogleModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(GoogleModelName))
                case Provider.VERTEXAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = VertexAIModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(VertexAIModelName))
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                case Provider.AWS:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                    self.AVAILABLE_MODELS.update(set(AWSModelName))
                case Provider.OLLAMA:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OllamaModelName.OLLAMA_GENERIC
                    self.AVAILABLE_MODELS.update(set(OllamaModelName))
                case Provider.OPENROUTER:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenRouterModelName.GEMINI_25_FLASH
                    self.AVAILABLE_MODELS.update(set(OpenRouterModelName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case Provider.AZURE_OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AzureOpenAIModelName.AZURE_GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(AzureOpenAIModelName))
                    # Validate Azure OpenAI settings if Azure provider is available
                    if not self.AZURE_OPENAI_API_KEY:
                        raise ValueError("AZURE_OPENAI_API_KEY must be set")
                    if not self.AZURE_OPENAI_ENDPOINT:
                        raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
                    if not self.AZURE_OPENAI_DEPLOYMENT_MAP:
                        raise ValueError("AZURE_OPENAI_DEPLOYMENT_MAP must be set")

                    # Parse deployment map if it's a string
                    if isinstance(self.AZURE_OPENAI_DEPLOYMENT_MAP, str):
                        try:
                            self.AZURE_OPENAI_DEPLOYMENT_MAP = loads(
                                self.AZURE_OPENAI_DEPLOYMENT_MAP
                            )
                        except Exception as e:
                            raise ValueError(f"Invalid AZURE_OPENAI_DEPLOYMENT_MAP JSON: {e}")

                    # Validate required deployments exist
                    required_models = {"gpt-4o", "gpt-4o-mini"}
                    missing_models = required_models - set(self.AZURE_OPENAI_DEPLOYMENT_MAP.keys())
                    if missing_models:
                        raise ValueError(f"Missing required Azure deployments: {missing_models}")
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\core\__init__.py

from core.llm import get_model
from core.settings import settings

__all__ = ["settings", "get_model"]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\memory\mongodb.py

import logging
import urllib.parse
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from core.settings import settings

logger = logging.getLogger(__name__)


def _has_auth_credentials() -> bool:
    required_auth = ["MONGO_USER", "MONGO_PASSWORD", "MONGO_AUTH_SOURCE"]
    set_auth = [var for var in required_auth if getattr(settings, var, None)]
    if len(set_auth) > 0 and len(set_auth) != len(required_auth):
        raise ValueError(
            f"If any of the following environment variables are set, all must be set: {', '.join(required_auth)}."
        )
    return len(set_auth) == len(required_auth)


def validate_mongo_config() -> None:
    """
    Validate that all required MongoDB configuration is present.
    Raises ValueError if any required configuration is missing.
    """
    required_always = ["MONGO_HOST", "MONGO_PORT", "MONGO_DB"]
    missing_always = [var for var in required_always if not getattr(settings, var, None)]
    if missing_always:
        raise ValueError(
            f"Missing required MongoDB configuration: {', '.join(missing_always)}. "
            "These environment variables must be set to use MongoDB persistence."
        )

    _has_auth_credentials()


def get_mongo_connection_string() -> str:
    """Build and return the MongoDB connection string from settings."""

    if _has_auth_credentials():
        if settings.MONGO_PASSWORD is None:  # for type checking
            raise ValueError("MONGO_PASSWORD is not set")
        password = settings.MONGO_PASSWORD.get_secret_value().strip()
        password_escaped = urllib.parse.quote_plus(password)
        return (
            f"mongodb://{settings.MONGO_USER}:{password_escaped}@"
            f"{settings.MONGO_HOST}:{settings.MONGO_PORT}/"
            f"?authSource={settings.MONGO_AUTH_SOURCE}"
        )
    else:
        return f"mongodb://{settings.MONGO_HOST}:{settings.MONGO_PORT}/"


def get_mongo_saver() -> AbstractAsyncContextManager[AsyncMongoDBSaver]:
    """Initialize and return a MongoDB saver instance."""
    validate_mongo_config()
    if settings.MONGO_DB is None:  # for type checking
        raise ValueError("MONGO_DB is not set")
    return AsyncMongoDBSaver.from_conn_string(
        get_mongo_connection_string(), db_name=settings.MONGO_DB
    )

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\memory\postgres.py

import logging
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from core.settings import settings

logger = logging.getLogger(__name__)


def validate_postgres_config() -> None:
    """
    Validate that all required PostgreSQL configuration is present.
    Raises ValueError if any required configuration is missing.
    """
    # required_vars = [
    #     "POSTGRES_USER",
    #     "POSTGRES_PASSWORD",
    #     "POSTGRES_HOST",
    #     "POSTGRES_PORT",
    #     "POSTGRES_DB",
    # ]
    required_vars = ["PGVECTOR_URL"]

    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "These environment variables must be set to use PostgreSQL persistence."
        )

    if settings.POSTGRES_MIN_CONNECTIONS_PER_POOL > settings.POSTGRES_MAX_CONNECTIONS_PER_POOL:
        raise ValueError(
            f"POSTGRES_MIN_CONNECTIONS_PER_POOL ({settings.POSTGRES_MIN_CONNECTIONS_PER_POOL}) must be less than or equal to POSTGRES_MAX_CONNECTIONS_PER_POOL ({settings.POSTGRES_MAX_CONNECTIONS_PER_POOL})"
        )


def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from settings."""
    if settings.PGVECTOR_URL is None:
        raise ValueError("PGVECTOR_URL is not set")
    return settings.PGVECTOR_URL


@asynccontextmanager
async def get_postgres_saver():
    """Initialize and return a PostgreSQL saver instance based on a connection pool for more resilent connections."""
    validate_postgres_config()
    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "saver"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        # Langgraph requires autocommmit=true and row_factory to be set to dict_row.
        # Application_name is passed so you can identify the connection in your Postgres database connection manager.
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        # makes sure that the connection is still valid before using it
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        try:
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            yield checkpointer
        finally:
            await pool.close()


@asynccontextmanager
async def get_postgres_store():
    """
    Get a PostgreSQL store instance based on a connection pool for more resilent connections.

    Returns an AsyncPostgresStore instance that can be used with async context manager pattern.

    """
    validate_postgres_config()
    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "store"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        # Langgraph requires autocommmit=true and row_factory to be set to dict_row
        # Application_name is passed so you can identify the connection in your Postgres database connection manager.
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        # makes sure that the connection is still valid before using it
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        try:
            store = AsyncPostgresStore(pool)
            await store.setup()
            yield store
        finally:
            await pool.close()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\memory\sqlite.py

from contextlib import AbstractAsyncContextManager, asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from core.settings import settings


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """Initialize and return a SQLite saver instance."""
    return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)


class AsyncInMemoryStore:
    """Wrapper for InMemoryStore that provides an async context manager interface."""

    def __init__(self):
        self.store = InMemoryStore()

    async def __aenter__(self):
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed for InMemoryStore
        pass

    async def setup(self):
        # No-op method for compatibility with PostgresStore
        pass


@asynccontextmanager
async def get_sqlite_store():
    """Initialize and return a store instance for long-term memory.

    Note: SQLite-specific store isn't available in LangGraph,
    so we use InMemoryStore wrapped in an async context manager for compatibility.
    """
    store_manager = AsyncInMemoryStore()
    yield await store_manager.__aenter__()

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\memory\__init__.py

from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.settings import DatabaseType, settings
from memory.mongodb import get_mongo_saver
from memory.postgres import get_postgres_saver, get_postgres_store
from memory.sqlite import get_sqlite_saver, get_sqlite_store


def initialize_database() -> AbstractAsyncContextManager[
    AsyncSqliteSaver | AsyncPostgresSaver | AsyncMongoDBSaver
]:
    """
    Initialize the appropriate database checkpointer based on configuration.
    Returns an initialized AsyncCheckpointer instance.
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_saver()
    if settings.DATABASE_TYPE == DatabaseType.MONGO:
        return get_mongo_saver()
    else:  # Default to SQLite
        return get_sqlite_saver()


def initialize_store():
    """
    Initialize the appropriate store based on configuration.
    Returns an async context manager for the initialized store.
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_store()
    # TODO: Add Mongo store - https://pypi.org/project/langgraph-store-mongodb/
    else:  # Default to SQLite
        return get_sqlite_store()


__all__ = ["initialize_database", "initialize_store"]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\schema\models.py

from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    OPENAI = auto()
    OPENAI_COMPATIBLE = auto()
    AZURE_OPENAI = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    VERTEXAI = auto()
    GROQ = auto()
    AWS = auto()
    OLLAMA = auto()
    OPENROUTER = auto()
    FAKE = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "1gpt-4o-mini"
    GPT_4O = "1gpt-4o"


class AzureOpenAIModelName(StrEnum):
    """Azure OpenAI model names"""

    AZURE_GPT_4O = "gpt-4o"
    AZURE_GPT_4O_MINI = "gpt-4o-mini"


class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""

    DEEPSEEK_CHAT = "deepseek-chat"


class AnthropicModelName(StrEnum):
    """https://docs.anthropic.com/en/docs/about-claude/models#model-names"""

    HAIKU_3 = "claude-3-haiku"
    HAIKU_35 = "claude-3.5-haiku"
    SONNET_35 = "claude-3.5-sonnet"


class GoogleModelName(StrEnum):
    """https://ai.google.dev/gemini-api/docs/models/gemini"""

    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"


class VertexAIModelName(StrEnum):
    """https://cloud.google.com/vertex-ai/generative-ai/docs/models"""

    GEMINI_15_PRO = "gemini-1.5-pro"
    GEMINI_20_FLASH = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "models/gemini-2.0-flash-lite"
    GEMINI_25_FLASH = "models/gemini-2.5-flash"
    GEMINI_25_PRO = "gemini-2.5-pro"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    LLAMA_31_8B = "llama-3.1-8b"
    LLAMA_33_70B = "llama-3.3-70b"

    LLAMA_GUARD_4_12B = "meta-llama/llama-guard-4-12b"


class AWSModelName(StrEnum):
    """https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"""

    BEDROCK_HAIKU = "bedrock-3.5-haiku"
    BEDROCK_SONNET = "bedrock-3.5-sonnet"


class OllamaModelName(StrEnum):
    """https://ollama.com/search"""

    OLLAMA_GENERIC = "ollama"


class OpenRouterModelName(StrEnum):
    """https://openrouter.ai/models"""

    GEMINI_25_FLASH = "google/gemini-2.5-flash"


class OpenAICompatibleName(StrEnum):
    """https://platform.openai.com/docs/guides/text-generation"""

    OPENAI_COMPATIBLE = "openai-compatible"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


AllModelEnum: TypeAlias = (
    OpenAIModelName
    | OpenAICompatibleName
    | AzureOpenAIModelName
    | DeepseekModelName
    | AnthropicModelName
    | GoogleModelName
    | VertexAIModelName
    | GroqModelName
    | AWSModelName
    | OllamaModelName
    | OpenRouterModelName
    | FakeModelName
)

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\schema\schema.py

from typing import Any, Literal, NotRequired

from pydantic import BaseModel, Field, SerializeAsAny
from typing_extensions import TypedDict

from schema.models import AllModelEnum, AnthropicModelName, OpenAIModelName


class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["research-assistant"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant for generating research papers."],
    )


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    models: list[AllModelEnum] = Field(
        description="List of available LLMs.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["research-assistant"],
    )
    default_model: AllModelEnum = Field(
        description="Default model used when none is specified.",
    )


class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="Model",
        description="LLM Model to use for the agent.",
        default=OpenAIModelName.GPT_4O_MINI,
        examples=[OpenAIModelName.GPT_4O_MINI, AnthropicModelName.HAIKU_35],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: str | None = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class Feedback(BaseModel):  # type: ignore[no-redef]
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class ChatHistory(BaseModel):
    messages: list[ChatMessage]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\schema\task_data.py

from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskData(BaseModel):
    name: str | None = Field(
        description="Name of the task.", default=None, examples=["Check input safety"]
    )
    run_id: str = Field(
        description="ID of the task run to pair state updates to.",
        default="",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    state: Literal["new", "running", "complete"] | None = Field(
        description="Current state of given task instance.",
        default=None,
        examples=["running"],
    )
    result: Literal["success", "error"] | None = Field(
        description="Result of given task instance.",
        default=None,
        examples=["running"],
    )
    data: dict[str, Any] = Field(
        description="Additional data generated by the task.",
        default={},
    )

    def completed(self) -> bool:
        return self.state == "complete"

    def completed_with_error(self) -> bool:
        return self.state == "complete" and self.result == "error"


class TaskDataStatus:
    def __init__(self) -> None:
        import streamlit as st

        self.status = st.status("")
        self.current_task_data: dict[str, TaskData] = {}

    def add_and_draw_task_data(self, task_data: TaskData) -> None:
        status = self.status
        status_str = f"Task **{task_data.name}** "
        match task_data.state:
            case "new":
                status_str += "has :blue[started]. Input:"
            case "running":
                status_str += "wrote:"
            case "complete":
                if task_data.result == "success":
                    status_str += ":green[completed successfully]. Output:"
                else:
                    status_str += ":red[ended with error]. Output:"
        status.write(status_str)
        status.write(task_data.data)
        status.write("---")
        if task_data.run_id not in self.current_task_data:
            # Status label always shows the last newly started task
            status.update(label=f"""Task: {task_data.name}""")
        self.current_task_data[task_data.run_id] = task_data
        if all(entry.completed() for entry in self.current_task_data.values()):
            # Status is "error" if any task has errored
            if any(entry.completed_with_error() for entry in self.current_task_data.values()):
                state = "error"
            # Status is "complete" if all tasks have completed successfully
            else:
                state = "complete"
        # Status is "running" until all tasks have completed
        else:
            state = "running"
        status.update(state=state)  # type: ignore[arg-type]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\schema\__init__.py

from schema.models import AllModelEnum
from schema.schema import (
    AgentInfo,
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)

__all__ = [
    "AgentInfo",
    "AllModelEnum",
    "UserInput",
    "ChatMessage",
    "ServiceMetadata",
    "StreamInput",
    "Feedback",
    "FeedbackResponse",
    "ChatHistoryInput",
    "ChatHistory",
]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\service\service.py

import inspect
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.callback import CallbackHandler  # type: ignore[import-untyped]
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info
from core import settings
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

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer and store
    based on settings.
    """
    try:
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
router = APIRouter(dependencies=[Depends(verify_bearer)])


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


async def _handle_input(user_input: UserInput, agent: AgentGraph) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    user_id = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "model": user_input.model, "user_id": user_id}

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

    return kwargs, run_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
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
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent)

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
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if node == "supervisor":
                        # Get only the last ToolMessage since is it added by the
                        # langgraph lib and not actual AI output so it won't be an
                        # independent event
                        if isinstance(update_messages[-1], ToolMessage):
                            update_messages = [update_messages[-1]]
                        else:
                            update_messages = []

                    if node in ("research_expert", "math_expert","team_supervisor"):
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
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
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
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
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
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
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

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\service\utils.py

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

from schema import ChatMessage


def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """Create a ChatMessage from a LangChain message."""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=message.content[0],
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\src\service\__init__.py

from service.service import app
import truststore
truststore.inject_into_ssl()

__all__ = ["app"]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\scripts\create_chroma_db.py

import os
import shutil

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
import truststore
truststore.inject_into_ssl()


# Load environment variables from the .env file
load_dotenv()


def create_chroma_db(
    folder_path: str,
    db_name: str = "./chroma_db",
    delete_chroma_db: bool = True,
    chunk_size: int = 2000,
    overlap: int = 500,
):
    embeddings = AzureOpenAIEmbeddings(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        model=os.environ["AZURE_OPENAI_EMBEDDER"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

    # Initialize Chroma vector store
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=f"./{db_name}",
    )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Load document based on file extension
        # Add more loaders if required, i.e. JSONLoader, TxtLoader, etc.
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue  # Skip unsupported file types

        # Load and split document into chunks
        document = loader.load()
        chunks = text_splitter.split_documents(document)

        # Add chunks to Chroma vector store
        for chunk in chunks:
            chunk_id = chroma.add_documents([chunk])
            if chunk_id:
                print(f"Chunk added with ID: {chunk_id}")
            else:
                print("Failed to add chunk")

        print(f"Document {filename} added to database.")

    print(f"Vector database created and saved in {db_name}.")
    return chroma


if __name__ == "__main__":
    # Path to the folder containing the documents
    folder_path = "./data"

    # Create the Chroma database
    chroma = create_chroma_db(folder_path=folder_path)

    # Create retriever from the Chroma database
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # Perform a similarity search
    query = "What's my company's mission and values"
    similar_docs = retriever.invoke(query)

    # Display results
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n🔹 Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}")

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\scripts\create_pgvector_db.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import truststore

truststore.inject_into_ssl()
load_dotenv()

# Pull your full connection string from env
PG_URL = os.environ["PGVECTOR_URL"]

# (Optional) if you want to rebuild from scratch each time:
# PRE_DELETE = True
PRE_DELETE = True

# 1) Set up your embeddings
embeddings = AzureOpenAIEmbeddings(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    model=os.environ["AZURE_OPENAI_EMBEDDER"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

def create_pgvector_db(folder_path: str, collection_name: str = "pgvector_embeddings"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        if not (fname.endswith(".pdf") or fname.endswith(".docx")):
            continue

        loader = PyPDFLoader(path) if fname.endswith(".pdf") else Docx2txtLoader(path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)

        # This will create the extension, tables, and collection on first run
        PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            connection_string=PG_URL,
            collection_name=collection_name,
            pre_delete_collection=PRE_DELETE,  # drop+recreate if you need a fresh table
            use_jsonb=True,                   # recommended for metadata
        )
        print(f"Ingested {len(chunks)} chunks from {fname}")

    # Return a live PGVector instance for querying
    return PGVector(
        connection_string=PG_URL,
        embedding_function=embeddings,
        collection_name=collection_name,
        use_jsonb=True,
    )

if __name__ == "__main__":
    retriever = create_pgvector_db("./data")
    # Wrap in a LangChain retriever if you like:
    retriever = retriever.as_retriever(search_kwargs={"k": 3})

    # Test it:
    results = retriever.get_relevant_documents("mission and values")
    for doc in results:
        print("—", doc.page_content[:200].replace("\n", " "), "…")

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\conftest.py

import os
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    with patch.dict(os.environ, {}, clear=True):
        yield

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\app\conftest.py

from unittest.mock import patch

import pytest

from schema import AgentInfo, ServiceMetadata
from schema.models import OpenAIModelName


@pytest.fixture
def mock_agent_client(mock_env):
    """Fixture for creating a mock AgentClient with a clean environment."""

    mock_info = ServiceMetadata(
        default_agent="test-agent",
        agents=[
            AgentInfo(key="test-agent", description="Test agent"),
            AgentInfo(key="chatbot", description="Chatbot"),
        ],
        default_model=OpenAIModelName.GPT_4O,
        models=[OpenAIModelName.GPT_4O, OpenAIModelName.GPT_4O_MINI],
    )

    with patch("client.AgentClient") as mock_agent_client:
        mock_agent_client_instance = mock_agent_client.return_value
        mock_agent_client_instance.info = mock_info
        yield mock_agent_client_instance

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\app\test_streamlit_app.py

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest
from streamlit.testing.v1 import AppTest

from client import AgentClientError
from schema import ChatHistory, ChatMessage
from schema.models import OpenAIModelName


def test_app_simple_non_streaming(mock_agent_client):
    """Test the full app - happy path"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    WELCOME_START = "Hello! I'm an AI agent. Ask me anything!"
    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    mock_agent_client.ainvoke = AsyncMock(
        return_value=ChatMessage(type="ai", content=RESPONSE),
    )

    assert at.chat_message[0].avatar == "assistant"
    assert at.chat_message[0].markdown[0].value.startswith(WELCOME_START)

    at.sidebar.toggle[0].set_value(False)  # Use Streaming = False
    at.chat_input[0].set_value(PROMPT).run()
    print(at)
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == RESPONSE
    assert not at.exception


def test_app_settings(mock_agent_client):
    """Test the full app - happy path"""
    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["user_id"] = "1234"
    at.run()

    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    mock_agent_client.ainvoke = AsyncMock(
        return_value=ChatMessage(type="ai", content=RESPONSE),
    )

    at.sidebar.toggle[0].set_value(False)  # Use Streaming = False
    assert at.sidebar.selectbox[0].value == "gpt-4o"
    assert mock_agent_client.agent == "test-agent"
    at.sidebar.selectbox[0].set_value("gpt-4o-mini")
    at.sidebar.selectbox[1].set_value("chatbot")
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    # Basic checks
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == RESPONSE

    # Check the args match the settings
    assert mock_agent_client.agent == "chatbot"
    mock_agent_client.ainvoke.assert_called_with(
        message=PROMPT,
        model=OpenAIModelName.GPT_4O_MINI,
        thread_id=at.session_state.thread_id,
        user_id="1234",
    )
    assert not at.exception


def test_app_thread_id_history(mock_agent_client):
    """Test the thread_id is generated"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # Reset and set thread_id
    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["thread_id"] = "1234"
    HISTORY = [
        ChatMessage(type="human", content="What is the weather?"),
        ChatMessage(type="ai", content="The weather is sunny."),
    ]
    mock_agent_client.get_history.return_value = ChatHistory(messages=HISTORY)
    at.run()
    print(at)
    assert at.session_state.thread_id == "1234"
    mock_agent_client.get_history.assert_called_with(thread_id="1234")
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == "What is the weather?"
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == "The weather is sunny."
    assert not at.exception


def test_app_feedback(mock_agent_client):
    """TODO: Can't figure out how to interact with st.feedback"""

    pass


@pytest.mark.asyncio
async def test_app_streaming(mock_agent_client):
    """Test the app with streaming enabled - including tool messages"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # Setup mock streaming response
    PROMPT = "What is 6 * 7?"
    ai_with_tool = ChatMessage(
        type="ai",
        content="",
        tool_calls=[{"name": "calculator", "id": "test_call_id", "args": {"expression": "6 * 7"}}],
    )
    tool_message = ChatMessage(type="tool", content="42", tool_call_id="test_call_id")
    final_ai_message = ChatMessage(type="ai", content="The answer is 42")

    messages = [ai_with_tool, tool_message, final_ai_message]

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)  # Use Streaming = True
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    response = at.chat_message[1]
    tool_status = response.status[0]
    assert response.avatar == "assistant"
    assert tool_status.label == "Tool Call: calculator"
    assert tool_status.icon == ":material/check:"
    assert tool_status.markdown[0].value == "Input:"
    assert tool_status.json[0].value == '{"expression": "6 * 7"}'
    assert tool_status.markdown[1].value == "Output:"
    assert tool_status.markdown[2].value == "42"
    assert response.markdown[-1].value == "The answer is 42"
    assert not at.exception


@pytest.mark.asyncio
async def test_app_init_error(mock_agent_client):
    """Test the app with an error in the agent initialization"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # Setup mock streaming response
    PROMPT = "What is 6 * 7?"
    mock_agent_client.astream.side_effect = AgentClientError("Error connecting to agent")

    at.toggle[0].set_value(True)  # Use Streaming = True
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    assert at.chat_message[0].avatar == "assistant"
    assert at.chat_message[1].avatar == "user"
    assert at.chat_message[1].markdown[0].value == PROMPT
    assert at.error[0].value == "Error generating response: Error connecting to agent"
    assert not at.exception


def test_app_new_chat_btn(mock_agent_client):
    at = AppTest.from_file("../../src/streamlit_app.py").run()
    thread_id_a = at.session_state.thread_id

    at.sidebar.button[0].click().run()

    assert at.session_state.thread_id != thread_id_a
    assert not at.exception

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\client\conftest.py

import pytest

from client import AgentClient


@pytest.fixture
def agent_client(mock_env):
    """Fixture for creating a test client with a clean environment."""
    ac = AgentClient(base_url="http://test", get_info=False)
    ac.update_agent("test-agent", verify=False)
    return ac

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\client\test_client.py

import json
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import Request, Response

from client import AgentClient, AgentClientError
from schema import AgentInfo, ChatHistory, ChatMessage, ServiceMetadata
from schema.models import OpenAIModelName


def test_init(mock_env):
    """Test client initialization with different parameters."""
    # Test default values
    client = AgentClient(get_info=False)
    assert client.base_url == "http://0.0.0.0"
    assert client.timeout is None

    # Test custom values
    client = AgentClient(
        base_url="http://test",
        timeout=30.0,
        get_info=False,
    )
    assert client.base_url == "http://test"
    assert client.timeout == 30.0
    client.update_agent("test-agent", verify=False)
    assert client.agent == "test-agent"


def test_headers(mock_env):
    """Test header generation with and without auth."""
    # Test without auth
    client = AgentClient(get_info=False)
    assert client._headers == {}

    # Test with auth
    with patch.dict(os.environ, {"AUTH_SECRET": "test-secret"}, clear=True):
        client = AgentClient(get_info=False)
        assert client._headers == {"Authorization": "Bearer test-secret"}


def test_invoke(agent_client):
    """Test synchronous invocation."""
    QUESTION = "What is the weather?"
    ANSWER = "The weather is sunny."

    # Mock successful response
    mock_request = Request("POST", "http://test/invoke")
    mock_response = Response(
        200,
        json={"type": "ai", "content": ANSWER},
        request=mock_request,
    )
    with patch("httpx.post", return_value=mock_response):
        response = agent_client.invoke(QUESTION)
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER

    # Test with model and thread_id
    with patch("httpx.post", return_value=mock_response) as mock_post:
        response = agent_client.invoke(
            QUESTION,
            model="gpt-4o",
            thread_id="test-thread",
        )
        assert isinstance(response, ChatMessage)
        # Verify request
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["message"] == QUESTION
        assert kwargs["json"]["model"] == "gpt-4o"
        assert kwargs["json"]["thread_id"] == "test-thread"

    # Test error response
    error_response = Response(500, text="Internal Server Error", request=mock_request)
    with patch("httpx.post", return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            agent_client.invoke(QUESTION)
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_ainvoke(agent_client):
    """Test asynchronous invocation."""
    QUESTION = "What is the weather?"
    ANSWER = "The weather is sunny."

    # Test successful response
    mock_request = Request("POST", "http://test/invoke")
    mock_response = Response(200, json={"type": "ai", "content": ANSWER}, request=mock_request)
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        response = await agent_client.ainvoke(QUESTION)
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER

    # Test with model and thread_id
    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        response = await agent_client.ainvoke(
            QUESTION,
            model="gpt-4o",
            thread_id="test-thread",
        )
        assert isinstance(response, ChatMessage)
        assert response.type == "ai"
        assert response.content == ANSWER
        # Verify request
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["message"] == QUESTION
        assert kwargs["json"]["model"] == "gpt-4o"
        assert kwargs["json"]["thread_id"] == "test-thread"

    # Test error response
    error_response = Response(500, text="Internal Server Error", request=mock_request)
    with patch("httpx.AsyncClient.post", return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await agent_client.ainvoke(QUESTION)
        assert "500 Internal Server Error" in str(exc.value)


def test_stream(agent_client):
    """Test synchronous streaming."""
    QUESTION = "What is the weather?"
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    # Create mock response with streaming events
    events = (
        [f"data: {json.dumps({'type': 'token', 'content': token})}" for token in TOKENS]
        + [
            f"data: {json.dumps({'type': 'message', 'content': {'type': 'ai', 'content': FINAL_ANSWER}})}"
        ]
        + ["data: [DONE]"]
    )

    # Mock the streaming response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = events
    mock_response.request = Request("POST", "http://test/stream")
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)

    with patch("httpx.stream", return_value=mock_response):
        # Collect all streamed responses
        responses = list(agent_client.stream(QUESTION))

        # Verify tokens were streamed
        assert len(responses) == len(TOKENS) + 1  # tokens + final message
        for i, token in enumerate(TOKENS):
            assert responses[i] == token

        # Verify final message
        final_message = responses[-1]
        assert isinstance(final_message, ChatMessage)
        assert final_message.type == "ai"
        assert final_message.content == FINAL_ANSWER

    # Test error response
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/stream")
    )
    error_response_mock = Mock()
    error_response_mock.__enter__ = Mock(return_value=error_response)
    error_response_mock.__exit__ = Mock(return_value=None)
    with patch("httpx.stream", return_value=error_response_mock):
        with pytest.raises(AgentClientError) as exc:
            list(agent_client.stream(QUESTION))
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_astream(agent_client):
    """Test asynchronous streaming."""
    QUESTION = "What is the weather?"
    TOKENS = ["The", " weather", " is", " sunny", "."]
    FINAL_ANSWER = "The weather is sunny."

    # Create mock response with streaming events
    events = (
        [f"data: {json.dumps({'type': 'token', 'content': token})}" for token in TOKENS]
        + [
            f"data: {json.dumps({'type': 'message', 'content': {'type': 'ai', 'content': FINAL_ANSWER}})}"
        ]
        + ["data: [DONE]"]
    )

    # Create an async iterator for the events
    async def async_events():
        for event in events:
            yield event

    # Mock the streaming response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.request = Request("POST", "http://test/stream")
    mock_response.aiter_lines = Mock(return_value=async_events())
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.stream = Mock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        # Collect all streamed responses
        responses = []
        async for response in agent_client.astream(QUESTION):
            responses.append(response)

        # Verify tokens were streamed
        assert len(responses) == len(TOKENS) + 1  # tokens + final message
        for i, token in enumerate(TOKENS):
            assert responses[i] == token

        # Verify final message
        final_message = responses[-1]
        assert isinstance(final_message, ChatMessage)
        assert final_message.type == "ai"
        assert final_message.content == FINAL_ANSWER

    # Test error response
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/stream")
    )
    error_response_mock = AsyncMock()
    error_response_mock.__aenter__ = AsyncMock(return_value=error_response)

    mock_client.stream.return_value = error_response_mock

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(AgentClientError) as exc:
            async for _ in agent_client.astream(QUESTION):
                pass
        assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_acreate_feedback(agent_client):
    """Test asynchronous feedback creation."""
    RUN_ID = "test-run"
    KEY = "test-key"
    SCORE = 0.8
    KWARGS = {"comment": "Great response!"}

    # Test successful response
    mock_response = Response(200, json={}, request=Request("POST", "http://test/feedback"))
    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        await agent_client.acreate_feedback(RUN_ID, KEY, SCORE, KWARGS)
        # Verify request
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["run_id"] == RUN_ID
        assert kwargs["json"]["key"] == KEY
        assert kwargs["json"]["score"] == SCORE
        assert kwargs["json"]["kwargs"] == KWARGS

    # Test error response
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/feedback")
    )
    with patch("httpx.AsyncClient.post", return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            await agent_client.acreate_feedback(RUN_ID, KEY, SCORE)
        assert "500 Internal Server Error" in str(exc.value)


def test_get_history(agent_client):
    """Test chat history retrieval."""
    THREAD_ID = "test-thread"
    HISTORY = {
        "messages": [
            {"type": "human", "content": "What is the weather?"},
            {"type": "ai", "content": "The weather is sunny."},
        ]
    }

    # Mock successful response
    mock_response = Response(200, json=HISTORY, request=Request("POST", "http://test/history"))
    with patch("httpx.post", return_value=mock_response):
        history = agent_client.get_history(THREAD_ID)
        assert isinstance(history, ChatHistory)
        assert len(history.messages) == 2
        assert history.messages[0].type == "human"
        assert history.messages[1].type == "ai"

    # Test error response
    error_response = Response(
        500, text="Internal Server Error", request=Request("POST", "http://test/history")
    )
    with patch("httpx.post", return_value=error_response):
        with pytest.raises(AgentClientError) as exc:
            agent_client.get_history(THREAD_ID)
        assert "500 Internal Server Error" in str(exc.value)


def test_info(agent_client):
    assert agent_client.info is None
    assert agent_client.agent == "test-agent"

    # Mock info response
    test_info = ServiceMetadata(
        default_agent="custom-agent",
        agents=[AgentInfo(key="custom-agent", description="Custom agent")],
        default_model=OpenAIModelName.GPT_4O,
        models=[OpenAIModelName.GPT_4O, OpenAIModelName.GPT_4O_MINI],
    )
    test_response = Response(
        200, json=test_info.model_dump(), request=Request("GET", "http://test/info")
    )

    # Update an existing client with info
    with patch("httpx.get", return_value=test_response):
        agent_client.retrieve_info()

    assert agent_client.info == test_info
    assert agent_client.agent == "custom-agent"

    # Test invalid update_agent
    with pytest.raises(AgentClientError) as exc:
        agent_client.update_agent("unknown-agent")
    assert "Agent unknown-agent not found in available agents: custom-agent" in str(exc.value)

    # Test a fresh client with info
    with patch("httpx.get", return_value=test_response):
        agent_client = AgentClient(base_url="http://test")
    assert agent_client.info == test_info
    assert agent_client.agent == "custom-agent"

    # Test error on invoke if no agent set
    agent_client = AgentClient(base_url="http://test", get_info=False)
    with pytest.raises(AgentClientError) as exc:
        agent_client.invoke("test")
    assert "No agent selected. Use update_agent() to select an agent." in str(exc.value)

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\core\test_llm.py

import os
from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import FakeListChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from core.llm import get_model
from schema.models import (
    AnthropicModelName,
    FakeModelName,
    GroqModelName,
    OllamaModelName,
    OpenAIModelName,
)


def test_get_model_openai():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        model = get_model(OpenAIModelName.GPT_4O_MINI)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-4o-mini"
        assert model.temperature == 0.5
        assert model.streaming is True


def test_get_model_anthropic():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
        model = get_model(AnthropicModelName.HAIKU_3)
        assert isinstance(model, ChatAnthropic)
        assert model.model == "claude-3-haiku"
        assert model.temperature == 0.5
        assert model.streaming is True


def test_get_model_groq():
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
        model = get_model(GroqModelName.LLAMA_31_8B)
        assert isinstance(model, ChatGroq)
        assert model.model_name == "llama-3.1-8b"
        assert model.temperature == 0.5


def test_get_model_groq_guard():
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}):
        model = get_model(GroqModelName.LLAMA_GUARD_4_12B)
        assert isinstance(model, ChatGroq)
        assert model.model_name == "meta-llama/llama-guard-4-12b"
        assert model.temperature < 0.01


def test_get_model_ollama():
    with patch("core.settings.settings.OLLAMA_MODEL", "llama3.3"):
        model = get_model(OllamaModelName.OLLAMA_GENERIC)
        assert isinstance(model, ChatOllama)
        assert model.model == "llama3.3"
        assert model.temperature == 0.5


def test_get_model_fake():
    model = get_model(FakeModelName.FAKE)
    assert isinstance(model, FakeListChatModel)
    assert model.responses == ["This is a test response from the fake model."]


def test_get_model_invalid():
    with pytest.raises(ValueError, match="Unsupported model:"):
        # Using type: ignore since we're intentionally testing invalid input
        get_model("invalid_model")  # type: ignore

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\core\test_settings.py

import json
import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from core.settings import Settings, check_str_is_http
from schema.models import (
    AnthropicModelName,
    AzureOpenAIModelName,
    OpenAIModelName,
    VertexAIModelName,
)


def test_check_str_is_http():
    # Test valid HTTP URLs
    assert check_str_is_http("http://example.com/") == "http://example.com/"
    assert check_str_is_http("https://api.test.com/") == "https://api.test.com/"

    # Test invalid URLs
    with pytest.raises(ValidationError):
        check_str_is_http("not_a_url")
    with pytest.raises(ValidationError):
        check_str_is_http("ftp://invalid.com")


def test_settings_default_values():
    settings = Settings(_env_file=None)
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8080
    assert settings.USE_AWS_BEDROCK is False
    assert settings.USE_FAKE_MODEL is False


def test_settings_no_api_keys():
    # Test that settings raises error when no API keys are provided
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
            _ = Settings(_env_file=None)


def test_settings_with_openai_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_4O_MINI
        assert settings.AVAILABLE_MODELS == set(OpenAIModelName)


def test_settings_with_anthropic_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.ANTHROPIC_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == AnthropicModelName.HAIKU_3
        assert settings.AVAILABLE_MODELS == set(AnthropicModelName)


def test_settings_with_vertexai_credentials_file():
    with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.GOOGLE_APPLICATION_CREDENTIALS == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == VertexAIModelName.GEMINI_20_FLASH
        assert settings.AVAILABLE_MODELS == set(VertexAIModelName)


def test_settings_with_multiple_api_keys():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "ANTHROPIC_API_KEY": "test_anthropic_key",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_openai_key")
        assert settings.ANTHROPIC_API_KEY == SecretStr("test_anthropic_key")
        # When multiple providers are available, OpenAI should be the default
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_4O_MINI
        # Available models should include exactly all OpenAI and Anthropic models
        expected_models = set(OpenAIModelName)
        expected_models.update(set(AnthropicModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_base_url():
    settings = Settings(HOST="0.0.0.0", PORT=8000, _env_file=None)
    assert settings.BASE_URL == "http://0.0.0.0:8000"


def test_settings_is_dev():
    settings = Settings(MODE="dev", _env_file=None)
    assert settings.is_dev() is True

    settings = Settings(MODE="prod", _env_file=None)
    assert settings.is_dev() is False


def test_settings_with_azure_openai_key():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-4o": "deployment-1", "gpt-4o-mini": "deployment-2"}',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.AZURE_OPENAI_API_KEY.get_secret_value() == "test_key"
        assert settings.DEFAULT_MODEL == AzureOpenAIModelName.AZURE_GPT_4O_MINI
        assert settings.AVAILABLE_MODELS == set(AzureOpenAIModelName)


def test_settings_with_both_openai_and_azure():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "AZURE_OPENAI_API_KEY": "test_azure_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-4o": "deployment-1", "gpt-4o-mini": "deployment-2"}',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_openai_key")
        assert settings.AZURE_OPENAI_API_KEY == SecretStr("test_azure_key")
        # When multiple providers are available, OpenAI should be the default
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_4O_MINI
        # Available models should include both OpenAI and Azure OpenAI models
        expected_models = set(OpenAIModelName)
        expected_models.update(set(AzureOpenAIModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_azure_deployment_names():
    # Delete this test
    pass


def test_settings_azure_missing_deployment_names():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        },
        clear=True,
    ):
        with pytest.raises(ValidationError, match="AZURE_OPENAI_DEPLOYMENT_MAP must be set"):
            Settings(_env_file=None)


def test_settings_azure_deployment_map():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-4o": "deploy1", "gpt-4o-mini": "deploy2"}',
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.AZURE_OPENAI_DEPLOYMENT_MAP == {
            "gpt-4o": "deploy1",
            "gpt-4o-mini": "deploy2",
        }


def test_settings_azure_invalid_deployment_map():
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": '{"gpt-4o": "deploy1"}',  # Missing required model
        },
        clear=True,
    ):
        with pytest.raises(ValueError, match="Missing required Azure deployments"):
            Settings(_env_file=None)


def test_settings_azure_openai():
    """Test Azure OpenAI settings."""
    deployment_map = {"gpt-4o": "deployment1", "gpt-4o-mini": "deployment2"}
    with patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT_MAP": json.dumps(deployment_map),
        },
    ):
        settings = Settings(_env_file=None)
        assert settings.AZURE_OPENAI_API_KEY.get_secret_value() == "test-key"
        assert settings.AZURE_OPENAI_ENDPOINT == "https://test.openai.azure.com"
        assert settings.AZURE_OPENAI_DEPLOYMENT_MAP == deployment_map

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\integration\test_docker_e2e.py

import pytest
from streamlit.testing.v1 import AppTest

from client import AgentClient


@pytest.mark.docker
def test_service_with_fake_model():
    """Test the service using the fake model.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    client = AgentClient("http://0.0.0.0", agent="chatbot")
    response = client.invoke("Tell me a joke?", model="fake")
    assert response.type == "ai"
    assert response.content == "This is a test response from the fake model."


@pytest.mark.docker
def test_service_with_app():
    """Test the service using the app.

    This test requires the service container to be running with USE_FAKE_MODEL=true
    """
    at = AppTest.from_file("../../src/streamlit_app.py").run()
    assert at.chat_message[0].avatar == "assistant"
    welcome = at.chat_message[0].markdown[0].value
    assert welcome.startswith("Hello! I'm an AI-powered research assistant")
    assert not at.exception

    at.sidebar.selectbox[1].set_value("chatbot")
    at.chat_input[0].set_value("What is the weather in Tokyo?").run()
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == "What is the weather in Tokyo?"
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == "This is a test response from the fake model."
    assert not at.exception

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\service\conftest.py

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from service import app


@pytest.fixture
def test_client():
    """Fixture to create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent that can be configured for different test scenarios."""
    agent_mock = AsyncMock()
    agent_mock.ainvoke = AsyncMock(
        return_value=[("values", {"messages": [AIMessage(content="Test response")]})]
    )
    agent_mock.get_state = Mock()  # Default empty mock for get_state
    with patch("service.service.get_agent", Mock(return_value=agent_mock)):
        yield agent_mock


@pytest.fixture
def mock_settings(mock_env):
    """Fixture to ensure settings are clean for each test."""
    with patch("service.service.settings") as mock_settings:
        yield mock_settings


@pytest.fixture
def mock_httpx():
    """Patch httpx.stream and httpx.get to use our test client."""

    with TestClient(app) as client:

        def mock_stream(method: str, url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0", "")
            return client.stream(method, path, **kwargs)

        def mock_get(url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0", "")
            return client.get(path, **kwargs)

        with patch("httpx.stream", mock_stream):
            with patch("httpx.get", mock_get):
                yield

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\service\test_auth.py

from pydantic import SecretStr


def test_no_auth_secret(mock_settings, mock_agent, test_client):
    """Test that when AUTH_SECRET is not set, all requests are allowed"""
    mock_settings.AUTH_SECRET = None
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer any-token"},
    )
    assert response.status_code == 200

    # Should also work without any auth header
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 200


def test_auth_secret_correct(mock_settings, mock_agent, test_client):
    """Test that when AUTH_SECRET is set, requests with correct token are allowed"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 200


def test_auth_secret_incorrect(mock_settings, mock_agent, test_client):
    """Test that when AUTH_SECRET is set, requests with wrong token are rejected"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert response.status_code == 401

    # Should also reject requests with no auth header
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 401

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\service\test_service.py

import json
from unittest.mock import AsyncMock, patch

import langsmith
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.pregel.types import StateSnapshot
from langgraph.types import Interrupt

from agents.agent_registry import Agent
from schema import ChatHistory, ChatMessage, ServiceMetadata
from schema.models import OpenAIModelName


def test_invoke(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent(test_client, mock_agent) -> None:
    """Test that /invoke works with a custom agent_id path parameter."""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    CUSTOM_ANSWER = "The weather in Tokyo is sunny."
    DEFAULT_ANSWER = "This is from the default agent."

    # Create a separate mock for the default agent
    default_mock = AsyncMock()
    default_mock.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=DEFAULT_ANSWER)]})
    ]

    # Configure our custom mock agent
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=CUSTOM_ANSWER)]})]

    # Patch get_agent to return the correct agent based on the provided agent_id
    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return mock_agent
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(f"/{CUSTOM_AGENT}/invoke", json={"message": QUESTION})
        assert response.status_code == 200

        # Verify custom agent was called and default wasn't
        mock_agent.ainvoke.assert_awaited_once()
        default_mock.ainvoke.assert_not_awaited()

        input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
        assert input_message.content == QUESTION

        output = ChatMessage.model_validate(response.json())
        assert output.type == "ai"
        assert output.content == CUSTOM_ANSWER  # Verify we got the custom agent's response


def test_invoke_model_param(test_client, mock_agent) -> None:
    """Test that the model parameter is correctly passed to the agent."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_MODEL = "claude-3.5-sonnet"
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION, "model": CUSTOM_MODEL})
    assert response.status_code == 200

    # Verify the model was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["model"] == CUSTOM_MODEL

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify an invalid model throws a validation error
    INVALID_MODEL = "gpt-7-notreal"
    response = test_client.post("/invoke", json={"message": QUESTION, "model": INVALID_MODEL})
    assert response.status_code == 422


def test_invoke_custom_agent_config(test_client, mock_agent) -> None:
    """Test that the agent_config parameter is correctly passed to the agent."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_CONFIG = {"spicy_level": 0.1, "additional_param": "value_foo"}

    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": CUSTOM_CONFIG}
    )
    assert response.status_code == 200

    # Verify the agent_config was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["spicy_level"] == 0.1
    assert config["configurable"]["additional_param"] == "value_foo"

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify a reserved key in agent_config throws a validation error
    INVALID_CONFIG = {"model": "gpt-4o"}
    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": INVALID_CONFIG}
    )
    assert response.status_code == 422


def test_invoke_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    INTERRUPT = "Confirm weather check"
    mock_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=ANSWER)]}),
        ("updates", {"__interrupt__": [Interrupt(value=INTERRUPT)]}),
    ]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == INTERRUPT


@patch("service.service.LangsmithClient")
def test_feedback(mock_client: langsmith.Client, test_client) -> None:
    ls_instance = mock_client.return_value
    ls_instance.create_feedback.return_value = None
    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    ls_instance.create_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="human-feedback-stars",
        score=0.8,
    )


def test_history(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    user_question = HumanMessage(content=QUESTION)
    agent_response = AIMessage(content=ANSWER)
    mock_agent.get_state.return_value = StateSnapshot(
        values={"messages": [user_question, agent_response]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    response = test_client.post(
        "/history", json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"}
    )
    assert response.status_code == 200

    output = ChatHistory.model_validate(response.json())
    assert output.messages[0].type == "human"
    assert output.messages[0].content == QUESTION
    assert output.messages[1].type == "ai"
    assert output.messages[1].content == ANSWER


@pytest.mark.asyncio
async def test_stream(test_client, mock_agent) -> None:
    """Test streaming tokens and messages."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": True}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify streamed tokens
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == len(TOKENS)
        for i, msg in enumerate(token_messages):
            assert msg["content"] == TOKENS[i]

        # Verify final message
        final_messages = [msg for msg in messages if msg["type"] == "message"]
        assert len(final_messages) == 1
        assert final_messages[0]["content"]["content"] == FINAL_ANSWER
        assert final_messages[0]["content"]["type"] == "ai"


@pytest.mark.asyncio
async def test_stream_no_tokens(test_client, mock_agent) -> None:
    """Test streaming without tokens."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify no token messages
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == 0

        # Verify final message
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["content"]["content"] == FINAL_ANSWER
        assert messages[0]["content"]["type"] == "ai"


def test_stream_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    INTERRUPT = "Confirm weather check"
    # Configure mock to use our async iterator function
    events = [
        (
            "updates",
            {"__interrupt__": [Interrupt(value=INTERRUPT)]},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify interrupt message
        assert len(messages) == 1
        assert messages[0]["content"]["content"] == INTERRUPT
        assert messages[0]["content"]["type"] == "ai"


def test_info(test_client, mock_settings) -> None:
    """Test that /info returns the correct service metadata."""

    base_agent = Agent(description="A base agent.", graph=None)
    mock_settings.AUTH_SECRET = None
    mock_settings.DEFAULT_MODEL = OpenAIModelName.GPT_4O_MINI
    mock_settings.AVAILABLE_MODELS = {OpenAIModelName.GPT_4O_MINI, OpenAIModelName.GPT_4O}
    with patch.dict("agents.agents.agents", {"base-agent": base_agent}, clear=True):
        response = test_client.get("/info")
        assert response.status_code == 200
        output = ServiceMetadata.model_validate(response.json())

    assert output.default_agent == "research-assistant"
    assert len(output.agents) == 1
    assert output.agents[0].key == "base-agent"
    assert output.agents[0].description == "A base agent."

    assert output.default_model == OpenAIModelName.GPT_4O_MINI
    assert output.models == [OpenAIModelName.GPT_4O, OpenAIModelName.GPT_4O_MINI]

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\service\test_service_e2e.py

from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter

from agents.agent_registry import Agent
from agents.utils import CustomData
from client import AgentClient
from schema.schema import ChatMessage
from service.utils import langchain_to_chat_message

START_MESSAGE = CustomData(type="start", data={"key1": "value1", "key2": 123})

STATIC_MESSAGES = [
    AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="test_tool",
                args={"arg1": "value1"},
                id="test_call_id",
            ),
        ],
    ),
    ToolMessage(content="42", tool_call_id="test_call_id"),
    AIMessage(content="The answer is 42"),
    CustomData(type="end", data={"time": "end"}).to_langchain(),
]


EXPECTED_OUTPUT_MESSAGES = [
    langchain_to_chat_message(m) for m in [START_MESSAGE.to_langchain()] + STATIC_MESSAGES
]


def test_messages_conversion() -> None:
    """Verify that our list of messages is converted to the expected output."""

    messages = EXPECTED_OUTPUT_MESSAGES

    # Verify the sequence of messages
    assert len(messages) == 5

    # First message: Custom data start marker
    assert messages[0].type == "custom"
    assert messages[0].custom_data == {"key1": "value1", "key2": 123}

    # Second message: AI with tool call
    assert messages[1].type == "ai"
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0]["name"] == "test_tool"
    assert messages[1].tool_calls[0]["args"] == {"arg1": "value1"}

    # Third message: Tool response
    assert messages[2].type == "tool"
    assert messages[2].content == "42"
    assert messages[2].tool_call_id == "test_call_id"

    # Fourth message: Final AI response
    assert messages[3].type == "ai"
    assert messages[3].content == "The answer is 42"

    # Fifth message: Custom data end marker
    assert messages[4].type == "custom"
    assert messages[4].custom_data == {"time": "end"}


async def static_messages(state: MessagesState, writer: StreamWriter) -> MessagesState:
    START_MESSAGE.dispatch(writer)
    return {"messages": STATIC_MESSAGES}


agent = StateGraph(MessagesState)
agent.add_node("static_messages", static_messages)
agent.set_entry_point("static_messages")
agent.add_edge("static_messages", END)
static_agent = agent.compile(checkpointer=MemorySaver())


def test_agent_stream(mock_httpx):
    """Test that streaming from our static agent works correctly with token streaming."""
    agent_meta = Agent(description="A static agent.", graph=static_agent)
    with patch.dict("agents.agents.agents", {"static-agent": agent_meta}, clear=True):
        client = AgentClient(agent="static-agent")

    # Use stream to get intermediate responses
    messages = []

    def agent_lookup(agent_id):
        if agent_id == "static-agent":
            return static_agent
        return None

    with patch("service.service.get_agent", side_effect=agent_lookup):
        for response in client.stream("Test message", stream_tokens=False):
            if isinstance(response, ChatMessage):
                messages.append(response)

    for expected, actual in zip(EXPECTED_OUTPUT_MESSAGES, messages):
        actual.run_id = None
        assert expected == actual

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\service\test_service_streaming.py

import pytest
from langchain_core.messages import AIMessage

from service.service import _create_ai_message


@pytest.mark.parametrize(
    "parts, expected",
    [
        # 1) Basic content + tool_calls
        (
            {"content": "Hello", "tool_calls": []},
            {"content": "Hello", "tool_calls": []},
        ),
        # 2) Unknown keys are ignored
        (
            {"content": "Test", "foobar": 123, "tool_calls": []},
            {"content": "Test", "tool_calls": []},
        ),
        # 3) Extra valid AIMessage params (id, type) pass through
        (
            {
                "content": "Hey",
                "id": "abc-123",
                "type": "ai",
                "tool_calls": [],
            },
            {"content": "Hey", "id": "abc-123", "type": "ai", "tool_calls": []},
        ),
    ],
)
def test_create_ai_message_filters_and_passes_through(parts, expected):
    """
    _create_ai_message should:
      - Drop unknown keys ("foobar")
      - Preserve keys that match AIMessage signature
      - Use the final value for duplicate keys in the parts dict
    """
    msg: AIMessage = _create_ai_message(parts)
    for key, val in expected.items():
        assert getattr(msg, key) == val


def test_create_ai_message_missing_required_content_raises():
    """
    AIMessage requires 'content'; if missing, _create_ai_message should
    bubble up the TypeError from the constructor.
    """
    with pytest.raises(TypeError):
        _create_ai_message({"tool_calls": []})


def test_create_ai_message_empty_dict_raises():
    """
    Completely empty parts should also fail to construct an AIMessage.
    """
    with pytest.raises(TypeError):
        _create_ai_message({})

```

```python
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\tests\service\test_utils.py

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage

from service.utils import langchain_to_chat_message


def test_messages_from_langchain() -> None:
    lc_human_message = HumanMessage(content="Hello, world!")
    human_message = langchain_to_chat_message(lc_human_message)
    assert human_message.type == "human"
    assert human_message.content == "Hello, world!"

    lc_ai_message = AIMessage(content="Hello, world!")
    ai_message = langchain_to_chat_message(lc_ai_message)
    assert ai_message.type == "ai"
    assert ai_message.content == "Hello, world!"

    lc_tool_message = ToolMessage(content="Hello, world!", tool_call_id="123")
    tool_message = langchain_to_chat_message(lc_tool_message)
    assert tool_message.type == "tool"
    assert tool_message.content == "Hello, world!"
    assert tool_message.tool_call_id == "123"

    lc_system_message = SystemMessage(content="Hello, world!")
    try:
        _ = langchain_to_chat_message(lc_system_message)
    except ValueError as e:
        assert str(e) == "Unsupported message type: SystemMessage"


def test_message_run_id_usage() -> None:
    run_id = "847c6285-8fc9-4560-a83f-4e6285809254"
    lc_message = AIMessage(content="Hello, world!")
    ai_message = langchain_to_chat_message(lc_message)
    ai_message.run_id = run_id
    assert ai_message.run_id == run_id


def test_messages_tool_calls() -> None:
    tool_call = ToolCall(name="test_tool", args={"x": 1, "y": 2}, id="call_Jja7")
    lc_ai_message = AIMessage(content="", tool_calls=[tool_call])
    ai_message = langchain_to_chat_message(lc_ai_message)
    assert ai_message.tool_calls[0]["id"] == "call_Jja7"
    assert ai_message.tool_calls[0]["name"] == "test_tool"
    assert ai_message.tool_calls[0]["args"] == {"x": 1, "y": 2}

```

```markdown
File: * C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\docs\File_Based_Credentials.md *

# File Based Crendentials 

As you develop your agents, you might discover that you have credentials that you need to store on disk that you don't want stored in your Git Repo or baked into your container image.

Examples:
- File based LLM Credentials Files (e.g. Google Vertex)
- Certificates or private keys needed for communication with external APIs


The `privatecredentials/` folder give you a quick place to put these files in development. 


## How it works

*Protection*
- The .dockerignore file excludes the entire folder to keep it out of the build process.  
- The .gitignore files only allows the `.gitkeep` file -- since git doesn't track empty folders.


*Mounted Volume*

The docker compose file mounts the `privatecredentials/` into the container as `/privatecredentials/`. The running container will have access to the untracked files that you have in your development environment.


*Why Not Use Docker Watch*

The syncing feature of Docker Watch isn't used for these reasons:
- docker watch adheres to the rules in `.dockerignore` and therefore won't see the credentials
- even if it did, docker watch doesn't do an initial sync when the container start and will only sync changes that occur while the service is running


## Suggested Use


For each file based credential, do the following:
1. Put the file (e.g. `example-creds.txt`) into the `privatecredentials/` folder
2. In your `.env` file, create an environment variable for the credential (e.g `EXAMPLE_CREDENTIAL=/privatecredentials/example-creds.txt`) that your agent will use to reference the location at runtime 
3. In your agent, use the environment variable wherever you need the path to the credential


### Examples

#### Google Vertex
Google Vertex SDK uses the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to locate your credentials file.

Do the following:
1. Put `service-account-key.json` (or `google-credentials.json`)  into the `privatecredentials/` folder
2. In your `.env` file, define `GOOGLE_APPLICATION_CREDENTIALS=/privatecredentials/service-account-key.json`
3. Vertex SDK automatically references the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.



#### Certificate For Signed Communication with remote API
If your agent calls a remote API that requires a client certificate, your agent will need the public certificate to be available.

For example, let's assume you have a cert named `my_remote_api_certificate.cer`

Do the following:
1. Put `my_remote_api_certificate.cer`  into the `privatecredentials/` folder
2. In your `.env` file, define `MY_REMOTE_API_CERTIFICATE=/privatecredentials/my_remote_api_certificate.cer`
3. Have the HTTP client in your agent access the file using the ENV value



## Production Options

In production, you will need to make the file based credentials available to the application and use the environment variable to define where the container can access them.

There are a number of approaches:

- Use Kubernetes Secrets or Docker Secrets mounted as data volumes that will let your app see them as files on the filesystem
- Use the secrets management feature of your cloud hosting environment (Google Cloud Secrets, AWS Secrets Manager, etc)
- Use a 3rd party secrets management platform
- Manually place the credentials on your Docker hosts and mount volumes to map the credentials to the container (Less secure)



```

```markdown
File: * C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\docs\Ollama.md *

# Using Ollama

⚠️ _**Note:** Ollama support in agent-service-toolkit is experimental and may not work as expected. The instructions below have been tested using Docker Desktop on a MacBook Pro. Please file an issue for any challenges you encounter._

You can also use [Ollama](https://ollama.com) to run the LLM powering the agent service.

1. Install Ollama using instructions from https://github.com/ollama/ollama
1. Install any model you want to use, e.g. `ollama pull llama3.2` and set the `OLLAMA_MODEL` environment variable to the model you want to use, e.g. `OLLAMA_MODEL=llama3.2`

If you are running the service locally (e.g. `python src/run_service.py`), you should be all set!

If you are running the service in Docker, you will also need to:

1. [Configure the Ollama server as described here](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server), e.g. by running `launchctl setenv OLLAMA_HOST "0.0.0.0"` on MacOS and restart Ollama.
1. Set the `OLLAMA_BASE_URL` environment variable to the base URL of the Ollama server, e.g. `OLLAMA_BASE_URL=http://host.docker.internal:11434`
1. Alternatively, you can run `ollama/ollama` image in Docker and use a similar configuration (however it may be slower in some cases).

```

```markdown
File: * C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\docs\RAG_Assistant.md *

# Creating a RAG assistant

You can build a RAG assistant using a Chroma database.

## Setting up Chroma

To create a Chroma database:

1. Add the data you want to use to a folder, i.e. `./data`, Word and PDF files are currently supported.
2. Open [`create_chroma_db.py` file](./scripts/create_chroma_db.py) and set the folder_path variable to the path to your data i.e. `./data`.
3. You can change the database name, chunk size and overlap size.
4. Assuming you have already followed the [Quickstart](#quickstart) and activated the virtual environment, to create the database run:

```sh
python scripts/create_chroma_db.py
```

5. If successful, a Chroma db will be created in the repository root directory.

## Configuring the RAG assistant

To create a RAG assistant:
1. Open [`tools.py` file](./src/agents/tools.py) and make sure the persist_directory is pointing to the database you created previously.
2. Modify the amount of documents returned, currently set to 5.
3. Update the `database_search_func` function description to accurately describe what the purpose and contents of your database is.
4. Open [`rag_assistant.py` file](./src/agents/rag_assistant.py) and update the agent's instuctions to describe what the assistant's speciality is and what knowledge it has access to, for example:

```python
instructions = f"""
    You are a helpful HR assistant with the ability to search a database containing information on our company's policies, benefits and handbook.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - If you have access to multiple databases, gather information from a diverse range of sources before crafting your response.
    - Please include the source of the information used in your response.
    - Use a friendly but professional tone when replying.
    - Only use information from the database. Do not use information from outside sources.
    """
```

5. Open [`streamlit_app.py` file](./src/streamlit_app.py) and update the agent's welcome message:

```python
WELCOME = """Hello! I'm your AI-powered HR assistant, here to help you navigate company policies, the employee handbook, and benefits. Ask me anything!""
```

6. Run the application and test your RAG assistant.

```

```markdown
File: * C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\docs\VertexAI.md *

# Working with Google Models

Google has two different ways for you to access their models. They are governed by different Terms of Use and usage policies.

Advising you on which method to use is beyond the scope of this document, but there are differences to consider.

The options are:

1. Gemini Developer API [link to documentation](https://ai.google.dev/gemini-api/docs)
2. Google Vertex AI on Google Cloud Platform [link to documentation](https://cloud.google.com/vertex-ai/docs)


## Using Gemini Developer API

[Get a Gemini API Key from Google](https://ai.google.dev/gemini-api/docs) and start using it quickly within the Agent Service Toolkit.

1. Put your API Key into the  `GOOGLE_API_KEY` environment variable in your `.env` file
2. Agent Service Toolkit should see the credentials and you should be good to go



## Using Google Vertex AI on Google Cloud Platform

### Prerequisites
Ensure you have a [Google Cloud project](https://console.cloud.google.com/projectcreate) with [billing enabled](https://console.cloud.google.com/billing).

### About Authentication
To use Vertex AI programmatically, you’ll create a **service account** and use its credentials to authenticate your application. These credentials, distinct from your personal Google Account credentials, determine your application’s access to Google Cloud services and APIs.

Vertex uses a JSON based credential file and read the `GOOGLE_APPLICATION_CREDENTIALS` Environment Variable to get the path to this credentials file at runtime.


### Models

Vertex AI includes both **stable** and **experimental/preview** models. Experimental and preview models may change or be discontinued without notice, so for **production applications**, it’s strongly recommended to use stable models. Check the [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs) for the latest information on model status.


### Steps

#### 1. Enable the Vertex AI API
- Go to the [Google Cloud API Library](https://console.cloud.google.com/apis/library).
- Select your project from the dropdown at the top.
- Search for "Vertex AI API" and click **Enable**.

#### 2. Create and Configure a Service Account
- Navigate to the [Credentials page](https://console.cloud.google.com/apis/credentials).
- Click **Create Credentials** > **Service Account**.
- Fill in the details (e.g., name and description).
- **Assign roles**: For Vertex AI, grant at least the "Vertex AI User" role.
- Click **Done**, then find your service account, click the three dots (⋮), and select **Manage Keys**.
- Click **Add Key** > **Create New Key**, select **JSON**, and click **Create**.
- The JSON key file will download. **Store it securely**—you won’t be able to download it again.

#### 3. Add the JSON Key File to the [File-based Credentials](docs/File_Based_Credentials.md) Path
Place the downloaded JSON file in the `privatecredentials/` of your project (e.g., `privatecredentials/service-account-key.json`).

Contents of the [File-based Credentials](docs/File_Based_Credentials.md) path are made available to your container at runtime in `/privatecredentials/` , but are excluded from git commits and docker builds.

#### 4. Set the `GOOGLE_APPLICATION_CREDENTIALS` Environment Variable
Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the **full path** of your JSON file:
  - **.env** (for docker compose) :
    ```
    GOOGLE_APPLICATION_CREDENTIALS=/privatecredentials/service-account-key.json
    ```
  - **Unix-like systems**:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/project/privatecredentials/service-account-key.json
    ```
  - **Windows**:
    ```cmd
    set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\project\privatecredentials\service-account-key.json
    ```

#### 5. Protect Your Credentials
- Make sure the JSON file is covered by your `.gitignore` file (Already done if you placed it in the provided `privatecredentials/` folder ) :
  ```
  service-account-key.json
  ```
- **Keep this file private**, as it grants access to your Google Cloud resources and could lead to **unauthorized usage** or billing.


### Verify Your Setup
Test your credentials with:
  ```bash
  gcloud auth activate-service-account --key-file=/path/to/your/service-account-key.json
  gcloud auth list
  ```
  Your service account should appear as active.

### Production Note
This setup is ideal for development. In production, consider more secure alternatives. Some options are listed on the [File-based Credentials](docs/File_Based_Credentials.md) page.

```

```DockerFile
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\.gitignore

/data/*

# Streamlit and sqlite
.streamlit/secrets.toml
checkpoints.db
checkpoints.db-*

# Langgraph
.langgraph_api/

# VSCode
.vscode
.DS_Store
*.code-workspace

# cursor
.cursorindexingignore
.specstory/

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/latest/usage/project/#working-with-version-control
.pdm.toml
.pdm-python
.pdm-build/

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.python-version
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
.idea/

# Data
*.docx
*.pdf

# Chroma db
*.db
*.sqlite3
*.bin

# Private Credentials, ignore everything in the folder but the .gitkeep file
privatecredentials/*
!privatecredentials/.gitkeep

```

```markdown
File: * C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\README.md *

# 🧰 AI Agent Service Toolkit

[![build status](https://github.drs.com/Steve-Long/Langgraph_Powered_Agent/actions/workflows/test.yml/badge.svg)](https://github.drs.com/Steve-Long/Langgraph_Powered_Agent/actions/workflows/test.yml)


A full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit.

It includes a [LangGraph](https://langchain-ai.github.io/langgraph/) agent, a [FastAPI](https://fastapi.tiangolo.com/) service to serve it, a client to interact with the service, and a [Streamlit](https://streamlit.io/) app that uses the client to provide a chat interface. Data structures and settings are built with [Pydantic](https://github.com/pydantic/pydantic).

This project offers a template for you to easily build and run your own agents using the LangGraph framework. It demonstrates a complete setup from agent definition to user interface, making it easier to get started with LangGraph-based projects by providing a full, robust toolkit.



## Overview

### [Try the app!](https://example.com/)

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Quickstart

Run directly in python

```sh
# At least one LLM API key is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# uv is the recommended way to install agent-service toolkit, but "pip install ." also works
# uv install options: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh

# Install dependencies. "uv sync" creates .venv automatically
uv sync --frozen
source .venv/bin/activate
python src/run_service.py

# In another shell
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

Run with docker

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

### Key Features

1. **LangGraph Agent and latest features**: A customizable agent built using the LangGraph framework. Implements the latest LangGraph v0.3 features including human in the loop with `interrupt()`, flow control with `Command`, long-term memory with `Store`, and `langgraph-supervisor`.
1. **FastAPI Service**: Serves the agent with both streaming and non-streaming endpoints.
1. **Advanced Streaming**: A novel approach to support both token-based and message-based streaming.
1. **Streamlit Interface**: Provides a user-friendly chat interface for interacting with the agent.
1. **Multiple Agent Support**: Run multiple agents in the service and call by URL path. Available agents and models are described in `/info`
1. **Asynchronous Design**: Utilizes async/await for efficient handling of concurrent requests.
1. **Content Moderation**: Implements LlamaGuard for content moderation (requires Groq API key).
1. **RAG Agent**: A basic RAG agent implementation using ChromaDB - see [docs](docs/RAG_Assistant.md).
1. **Feedback Mechanism**: Includes a star-based feedback system integrated with LangSmith.
1. **Docker Support**: Includes Dockerfiles and a docker compose file for easy development and deployment.
1. **Testing**: Includes robust unit and integration tests for the full repo.

### Key Files

The repository is structured as follows:

- `src/agents/`: Defines several agents with different capabilities
- `src/schema/`: Defines the protocol schema
- `src/core/`: Core modules including LLM definition and settings
- `src/service/service.py`: FastAPI service to serve the agents
- `src/client/client.py`: Client to interact with the agent service
- `src/streamlit_app.py`: Streamlit app providing a chat interface
- `tests/`: Unit and integration tests

## Setup and Usage

1. Clone the repository:

   ```sh
   git clone https://github.drs.com/Steve-Long/Langgraph_Powered_Agent.git
   cd agent-service-toolkit
   ```

2. Set up environment variables:
   Create a `.env` file in the root directory. At least one LLM API key or configuration is required. See the [`.env.example` file](./.env.example) for a full list of available environment variables, including a variety of model provider API keys, header-based authentication, LangSmith tracing, testing and development modes, and OpenWeatherMap API key.

3. You can now run the agent service and the Streamlit app locally, either with Docker or just using Python. The Docker setup is recommended for simpler environment setup and immediate reloading of the services when you make changes to your code.

### Additional setup for specific AI providers

- [Setting up Ollama](docs/Ollama.md)
- [Setting up VertexAI](docs/VertexAI.md)
- [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Building or customizing your own agent

To customize the agent for your own use case:

1. Add your new agent to the `src/agents` directory. You can copy `research_assistant.py` or `chatbot.py` and modify it to change the agent's behavior and tools.
1. Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`. Your agent can be called by `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
1. Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.


### Handling Private Credential files

If your agents or chosen LLM require file-based credential files or certificates, the `privatecredentials/` has been provided for your development convenience. All contents, excluding the `.gitkeep` files, are ignored by git and docker's build process. See [Working with File-based Credentials](docs/File_Based_Credentials.md) for suggested use.


### Docker Setup

This project includes a Docker setup for easy development and deployment. The `compose.yaml` file defines three services: `postgres`, `agent_service` and `streamlit_app`. The `Dockerfile` for each service is in their respective directories.

For local development, we recommend using [docker compose watch](https://docs.docker.com/compose/file-watch/). This feature allows for a smoother development experience by automatically updating your containers when changes are detected in your source code.

1. Make sure you have Docker and Docker Compose (>=[2.23.0](https://docs.docker.com/compose/release-notes/#2230)) installed on your system.

2. Create a `.env` file from the `.env.example`. At minimum, you need to provide an LLM API key (e.g., OPENAI_API_KEY).
   ```sh
   cp .env.example .env
   # Edit .env to add your API keys
   ```

3. Build and launch the services in watch mode:

   ```sh
   docker compose watch
   ```

   This will automatically:
   - Start a PostgreSQL database service that the agent service connects to
   - Start the agent service with FastAPI
   - Start the Streamlit app for the user interface

4. The services will now automatically update when you make changes to your code:
   - Changes in the relevant python files and directories will trigger updates for the relevant services.
   - NOTE: If you make changes to the `pyproject.toml` or `uv.lock` files, you will need to rebuild the services by running `docker compose up --build`.

5. Access the Streamlit app by navigating to `http://localhost:8501` in your web browser.

6. The agent service API will be available at `http://0.0.0.0:8080`. You can also use the OpenAPI docs at `http://0.0.0.0:8080/redoc`.

7. Use `docker compose down` to stop the services.

This setup allows you to develop and test your changes in real-time without manually restarting the services.

### Building other apps on the AgentClient

The repo includes a generic `src/client/client.AgentClient` that can be used to interact with the agent service. This client is designed to be flexible and can be used to build other apps on top of the agent. It supports both synchronous and asynchronous invocations, and streaming and non-streaming requests.

See the `src/run_client.py` file for full examples of how to use the `AgentClient`. A quick example:

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("Tell me a brief joke?")
response.pretty_print()
# ================================== Ai Message ==================================
#
# A man walked into a library and asked the librarian, "Do you have any books on Pavlov's dogs and Schrödinger's cat?"
# The librarian replied, "It rings a bell, but I'm not sure if it's here or not."

```

### Development with LangGraph Studio

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), the IDE for developing agents in LangGraph.

`langgraph-cli[inmem]` is installed with `uv sync`. You can simply add your `.env` file to the root directory as described above, and then launch LangGraph Studio with `langgraph dev`. Customize `langgraph.json` as needed. See the [local quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server) to learn more.

### Local development without Docker

You can also run the agent service and the Streamlit app locally without Docker, just using a Python virtual environment.

1. Create a virtual environment and install dependencies:

   ```sh
   uv sync --frozen
   . .\.venv\Scripts\Activate.ps1
   ```

2. Run the FastAPI server:

   ```sh
   python src/run_service.py
   ```

3. In a separate terminal, run the Streamlit app:

   ```sh
   streamlit run src/streamlit_app.py
   ```

4. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Projects built with or inspired by agent-service-toolkit

The following are a few of the public projects that drew code or inspiration from this repo.

- **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
- **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
- **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Please create a pull request editing the README or open a discussion with any new ones to be added!** Would love to include more projects.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Currently the tests need to be run using the local development without Docker setup. To run the tests for the agent service:

1. Ensure you're in the project root directory and have activated your virtual environment.

2. Install the development dependencies and pre-commit hooks:

   ```sh
   uv sync --frozen
   pre-commit install
   ```

3. Run the tests using pytest:

   ```sh
   pytest
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```

```YAML
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\compose.yaml

services:
  postgres:
    image: postgres:16
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=${POSTGRES_USER:-postgres}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres}
      - POSTGRES_DB=${POSTGRES_DB:-agent_service}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres}"]
      interval: 5s
      timeout: 5s
      retries: 5

  agent_service:
    build:
      context: .
      dockerfile: docker/Dockerfile.service
    ports:
      - "8080:8080"
    env_file:
      - .env
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/info"]
      interval: 5s
      timeout: 5s
      retries: 5
    # Volume mount of development-only credentials since compose watch doesn't sync ignored files and doesn't do initial sync when the service starts
    volumes:
      - ./privatecredentials:/privatecredentials
    develop:
      watch:
        - path: src/agents/
          action: sync+restart
          target: /app/agents/
        - path: src/schema/
          action: sync+restart
          target: /app/schema/
        - path: src/service/
          action: sync+restart
          target: /app/service/
        - path: src/core/
          action: sync+restart
          target: /app/core/
        - path: src/memory/
          action: sync+restart
          target: /app/memory/
  streamlit_app:
    build:
      context: .
      dockerfile: docker/Dockerfile.app
    ports:
      - "8501:8501"
    depends_on:
      - agent_service
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/healthz"]
      interval: 5s
      timeout: 5s
      retries: 5
    environment:
      - AGENT_URL=http://agent_service:8080
    develop:
      watch:
        - path: src/client/
          action: sync+restart
          target: /app/client/
        - path: src/schema/
          action: sync+restart
          target: /app/schema/
        - path: src/streamlit_app.py
          action: sync+restart
          target: /app/streamlit_app.py

volumes:
  postgres_data:

```

```YAML
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\.pre-commit-config.yaml

repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v2.3.0
  hooks:
    - id: check-yaml
    - id: end-of-file-fixer
    - id: trailing-whitespace
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.11.9
  hooks:
    - id: ruff
      args: [ --fix ]
    - id: ruff-format

```

```DockerFile
# C:\Users\Steve.Long\Downloads\agent-service-toolkit-main\agent-service-toolkit-main\.dockerignore

.git
.gitignore
.env
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
.venv
*.db
privatecredentials/*
```

