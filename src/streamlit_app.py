# src/streamlit_app.py

import asyncio
import logging
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import msal  # ← Added for Azure AD OAuth2
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from core.langfuse import build_trace_url, get_langfuse_client
from schema import ChatMessage
from schema.task_data import TaskData, TaskDataStatus

logger = logging.getLogger(__name__)

if os.getenv("ST_DEBUGPY", "0") == "1" and os.getenv("DEBUGPY_LAUNCHER_PORT") is None:
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()  # optional




# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:
#
# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.
#
# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.

APP_TITLE = "AIS Agent Platform"
APP_ICON = "🤖"
USER_ID_COOKIE = "user_id"
TRACE_ID_QUERY_PARAM = "trace_id"
DEBUG = False

# ——— OAuth2 Configuration ———
load_dotenv()
TENANT_ID = os.getenv("AZURE_AD_TENANT_ID")
CLIENT_ID = os.getenv("AZURE_AD_CLIENT_ID")
CLIENT_SECRET = os.getenv("AZURE_AD_CLIENT_SECRET")
# The redirect URI must match one of those registered in your Azure AD app
REDIRECT_URI = os.getenv("STREAMLIT_REDIRECT_URI")  # e.g. "https://your-streamlit-app.azurewebsites.net/"
AUTHORITY = f"https://login.microsoftonline.us/{TENANT_ID}"
# Scope should include your API's custom scope; replace with your actual scope URI
SCOPE = [f"api://{os.getenv('AZURE_AD_API_CLIENT_ID')}/access_as_user"]
msal_app = msal.ConfidentialClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
    client_credential=CLIENT_SECRET,
)

# ——— Auth Toggle ———
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

if os.getenv("ST_DEBUGPY","0") == "1":
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("🔎 debugpy listening on 5678")
    # debugpy.wait_for_client()  # uncomment to pause on start

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


def _get_query_param(name: str) -> str | None:
    """Return the first query parameter value for ``name`` if present."""

    value = st.query_params.get(name)
    if value is None:
        return None
    if isinstance(value, list):
        return value[0]
    return value

def render_payload(content: str) -> bool:
    """
    Detect and render a plotting payload produced by the coding agent.
    Returns True if handled, False to let caller fall back to st.write().
    Supported:
      - PLOTLY_JSON:<json>
      - DATA_URI:data:image/png;base64,...
    """
    if DEBUG:
        st.caption("🔎 render_payload called")
        st.code((content[:2000] if isinstance(content, str) else repr(content)) or "<empty>")

    if not isinstance(content, str) or not content:
        return False

    if content.startswith("PLOTLY_JSON:"):
        raw = content[len("PLOTLY_JSON:"):]
        if DEBUG:
            st.text(f"🔎 Detected PLOTLY_JSON (len={len(raw)})")
        try:
            fig = pio.from_json(raw)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Failed to parse Plotly JSON: {e}")
            if DEBUG:
                st.code(raw[:2000])
        return True

    if content.startswith("DATA_URI:"):
        uri = content[len("DATA_URI:"):]
        if DEBUG:
            st.text(f"🔎 Detected DATA_URI (len={len(uri)})")
        st.image(uri, use_column_width=True)
        return True

    if DEBUG:
        st.text("🔎 No payload marker matched")
    return False


def write_any(content: str, *, in_status=None):
    """
    Try to render a plot payload; otherwise write plain text.
    If in_status is provided (a status container), write inside it;
    but plots will render in a normal container just below for visibility.
    """
    # Prefer to render plots OUTSIDE status boxes (status can be collapsible/narrow)
    if isinstance(content, str) and (content.startswith("PLOTLY_JSON:") or content.startswith("DATA_URI:")):
        # Render below the current chat message (not inside status)
        holder = st.container()
        with holder:
            if not render_payload(content):
                st.write(content)
        # If you still want to show a short “Output:” line in the status:
        if in_status is not None:
            in_status.write("Output: (rendered below)")
        return

    # Non-plot text: write where it came from
    if in_status is not None:
        in_status.write(content)
    else:
        st.write(content)



def update_langfuse_runtime(agent_client: AgentClient | None = None) -> None:
    """Cache the current Langfuse runtime context in session state."""

    trace_id = st.session_state.get("trace_id")
    if not trace_id:
        st.session_state.langfuse_runtime = None
        return

    runtime: dict[str, Any] = {"trace_id": trace_id}
    thread_id = st.session_state.get("thread_id")
    if thread_id:
        runtime["session_id"] = thread_id
        runtime["thread_id"] = thread_id
    user_id = st.session_state.get(USER_ID_COOKIE)
    if user_id:
        runtime["user_id"] = user_id
    if agent_client and agent_client.agent:
        runtime["agent_name"] = agent_client.agent

    st.session_state.langfuse_runtime = runtime


def emit_frontend_event(name: str, metadata: dict[str, Any] | None = None) -> None:
    """Emit a Langfuse event scoped to the current frontend runtime."""

    try:
        client = get_langfuse_client()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Skipping frontend event %s; Langfuse client unavailable", name, exc_info=True)
        return

    if client is None:
        return

    trace_id = st.session_state.get("trace_id") or st.session_state.get("langfuse_trace_id")
    session_id = st.session_state.get("thread_id") or st.session_state.get("langfuse_session_id")
    user_id = st.session_state.get(USER_ID_COOKIE)

    event_payload: dict[str, Any] = {"name": name}
    if metadata:
        event_payload["metadata"] = metadata
    if trace_id:
        event_payload["trace_id"] = trace_id
    if session_id:
        event_payload["session_id"] = session_id
    if user_id:
        event_payload["user_id"] = user_id

    try:
        client.event(**event_payload)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Langfuse event %s failed; continuing without telemetry", name, exc_info=True)


def _store_stream_langfuse_metadata(message: ChatMessage) -> None:
    """Persist Langfuse identifiers from streamed messages into session state."""

    if not isinstance(message.custom_data, dict):
        return

    langfuse_meta = message.custom_data.get("langfuse")
    if not isinstance(langfuse_meta, dict):
        return

    changed = False

    trace_id = langfuse_meta.get("trace_id")
    if trace_id and st.session_state.get("langfuse_trace_id") != trace_id:
        st.session_state["langfuse_trace_id"] = trace_id
        changed = True

    session_id = langfuse_meta.get("session_id")
    if session_id and st.session_state.get("langfuse_session_id") != session_id:
        st.session_state["langfuse_session_id"] = session_id
        changed = True

    run_id = langfuse_meta.get("run_id") or message.run_id
    if run_id and st.session_state.get("langfuse_run_id") != run_id:
        st.session_state["langfuse_run_id"] = run_id
        changed = True

    user_id = langfuse_meta.get("user_id")
    if user_id and st.session_state.get("langfuse_user_id") != user_id:
        st.session_state["langfuse_user_id"] = user_id
        changed = True

    if changed:
        update_langfuse_runtime()


def render_trace_link() -> None:
    """Render a Langfuse trace hyperlink when configuration allows."""

    trace_url = build_trace_url(st.session_state.get("trace_id"))
    if not trace_url:
        return

    st.markdown(f"[🔗 View in Langfuse]({trace_url})")


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    ## Used to not display the 'oauth2callback' tab
    st.markdown("""
<style>
/* Hide oauth2callback from sidebar nav and top tab bar (covers both layouts) */
section[data-testid="stSidebar"] a[href$="/oauth2callback"],
section[data-testid="stSidebar"] a[href*="/oauth2callback?"],
header [data-testid="stAppTabBar"] a[href$="/oauth2callback"],
header [data-testid="stAppTabBar"] a[href*="/oauth2callback?"] { display: none !important; }
</style>
""", unsafe_allow_html=True)
    
    ## Used to not display the 'streamlit_app' tab
    st.markdown("""
<style>
/* Hide streamlit_app from sidebar nav and top tab bar (covers both layouts) */
section[data-testid="stSidebar"] a[href$="/streamlit_app"],
section[data-testid="stSidebar"] a[href*="/streamlit_app?"],
header [data-testid="stAppTabBar"] a[href$="/streamlit_app"],
header [data-testid="stAppTabBar"] a[href*="/streamlit_app?"] { display: none !important; }
</style>
""", unsafe_allow_html=True)    

    # ——— Handle OAuth2 redirect & token exchange ———
    if AUTH_ENABLED:
        params = st.query_params
        if "code" in params and "access_token" not in st.session_state:
            code = params["code"][0]
            result = msal_app.acquire_token_by_authorization_code(
                code,
                scopes=SCOPE,
                redirect_uri=REDIRECT_URI,
            )
            token = result.get("access_token")
            if not token:
                st.error("Failed to obtain access token.")
                return
            st.session_state["access_token"] = token
            # clear the code from URL
            st.experimental_set_query_params()

        # ——— Require login if we don’t yet have a token ———
        if "access_token" not in st.session_state:
            auth_url = msal_app.get_authorization_request_url(
                scopes=SCOPE,
                redirect_uri=REDIRECT_URI,
            )
            # st.markdown(f"[Sign in with Microsoft]({auth_url})")


            # auth_url = msal_app.get_authorization_request_url(
            #     scopes=SCOPE,
            #     redirect_uri=REDIRECT_URI,
            # )
            # st.html(f'<a href="{auth_url}" target="_self" rel="noopener">Sign in with Microsoft</a>')


            auth_url = msal_app.get_authorization_request_url(
                scopes=SCOPE,
                redirect_uri=REDIRECT_URI,
            )

            st.markdown(
                f"""
                <a href="{auth_url}" target="_self" style="
                    display: inline-block;
                    padding: 0.5em 1em;
                    background-color: #2e6ef7;
                    color: white;
                    border-radius: 4px;
                    text-decoration: none;
                    font-weight: bold;
                ">
                    Sign in with Microsoft
                </a>
                """,
                unsafe_allow_html=True
            )

            return  # stop further rendering until signed in

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
        agent_url = os.getenv("AGENT_URL") or f"http://{os.getenv('HOST','0.0.0.0')}:{os.getenv('PORT',8080)}"
        try:
            with st.spinner("Connecting to agent service..."):
                # Inject access_token only if auth is enabled; otherwise pass None
                client = AgentClient(
                    base_url=agent_url,
                    access_token=st.session_state.get("access_token") if AUTH_ENABLED else None
                )
                st.session_state.agent_client = client
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = _get_query_param("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    if "trace_id" not in st.session_state:
        trace_id = _get_query_param(TRACE_ID_QUERY_PARAM)
        if not trace_id:
            trace_id = str(uuid.uuid4())
        st.session_state.trace_id = trace_id
        st.query_params[TRACE_ID_QUERY_PARAM] = trace_id

    update_langfuse_runtime(agent_client)

    if not st.session_state.get("langfuse_page_load_event_sent"):
        emit_frontend_event(
            "frontend.page_load",
            {
                "app": "streamlit",
                "agent": agent_client.agent,
            },
        )
        st.session_state.langfuse_page_load_event_sent = True

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""
        DEBUG = st.toggle("🔎 Debug mode", value=False)
        globals()["DEBUG"] = DEBUG  # <— ADD THIS

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.trace_id = str(uuid.uuid4())
            st.query_params[TRACE_ID_QUERY_PARAM] = st.session_state.trace_id
            update_langfuse_runtime(agent_client)
            st.rerun()

        with st.popover(":material/settings: Agents/Workflows", use_container_width=True):
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
            file_path = f"./workflow_diagrams/{agent_client.agent}.png"
            if os.path.exists(file_path):
                st.image(file_path)
            else:
                st.warning("No diagram available for this agent.")


        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: TBD", use_container_width=True):
            st.write(
                "TBD"
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
                f"&{TRACE_ID_QUERY_PARAM}={st.session_state.trace_id}"
            )
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.drs.com/AIS-FWB-Engineering/AIS-Agent-Platform.git)"
        st.caption(
            "Made with :material/favorite: by [Steve](https://google.com) for you"
        )

        # --- Upload popover ---
        with st.popover(":material/upload_file: Upload files", use_container_width=True):
            st.caption("Attach files to this thread (auth required).")
            up_files = st.file_uploader(
                "Choose files",
                type=None,
                accept_multiple_files=True,
                key="file_uploader_multi",
                help="Documents, images, CSV, etc.",
            )
            do_ingest = st.toggle("Ingest after upload (if enabled server-side)", value=False, help="Requires server AUTO_INGEST_UPLOADS=True")
            if st.button("Upload", use_container_width=True, disabled=not up_files):
                if AUTH_ENABLED and "access_token" not in st.session_state:
                    st.error("Sign in first.")
                else:
                    results_placeholder = st.container()
                    with results_placeholder:
                        status_box = st.status("Uploading...", state="running")
                        # Build payload for client
                        payload: list[tuple[str, bytes, str | None]] = []
                        for f in up_files:
                            payload.append((f.name, f.getvalue(), f.type))
                        try:
                            resp = agent_client.upload_files(
                                payload,
                                thread_id=st.session_state.thread_id,
                                trace_id=st.session_state.get("trace_id")
                                or st.session_state.get("langfuse_trace_id"),
                                session_id=st.session_state.get("thread_id")
                                or st.session_state.get("langfuse_session_id"),
                                langfuse_run_id=st.session_state.get("langfuse_run_id"),
                            )
                            emit_frontend_event(
                                "frontend.file_upload",
                                {
                                    "count": len(payload),
                                    "ingest": do_ingest,
                                    "filenames": [name for name, *_ in payload],
                                },
                            )
                            # Render per-file outcomes
                            for item in resp:
                                state = item.get("status")
                                meta = item.get("file")
                                name = (meta or {}).get("original_name", "unknown")
                                match state:
                                    case "stored":
                                        status_box.write(f"✅ **{name}** — Saved")
                                        st.toast(f"Saved: {name}", icon="✅")
                                    case "skipped":
                                        status_box.write(f"⏭️ **{name}** — {item.get('message','Skipped')}")
                                        st.toast(f"Skipped: {name}", icon="⏭️")
                                    case "error":
                                        status_box.write(f"❌ **{name}** — {item.get('message','Error')}")
                                        st.toast(f"Failed: {name}", icon="❌")
                            status_box.update(label="Upload complete", state="complete")
                        except AgentClientError as e:
                            status_box.update(label="Upload failed", state="error")
                            emit_frontend_event(
                                "frontend.file_upload.error",
                                {
                                    "count": len(payload),
                                    "ingest": do_ingest,
                                    "error": str(e),
                                },
                            )
                            st.error(str(e))


    update_langfuse_runtime(agent_client)

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
            case "interactive-ideation-agent":
                WELCOME = """Hello! Tell me about your process improvement idea"""
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
                    trace_id=st.session_state.trace_id,
                    session_id=st.session_state.thread_id,
                    # user_id is no longer sent; identity comes from JWT on the server
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    trace_id=st.session_state.trace_id,
                    session_id=st.session_state.thread_id,
                    # user_id is no longer sent; identity comes from JWT on the server
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
            render_trace_link()
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
    streaming_is_payload = False 

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

            if DEBUG and len(streaming_content) < 300:
                with st.session_state.last_message:
                    st.text(f"🔎 token += … now {len(streaming_content)} chars")
                    st.code(streaming_content)

            if not streaming_is_payload and (
                streaming_content.startswith("PLOTLY_JSON:") or streaming_content.startswith("DATA_URI:")
            ):
                streaming_is_payload = True

                if DEBUG:
                    with st.session_state.last_message:
                        st.text("🔎 Detected payload marker during streaming")

            # Only show tokens if it's normal text; hide when it's a plot payload
            if not streaming_is_payload:
                streaming_placeholder.write(streaming_content)

            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        _store_stream_langfuse_metadata(msg)

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
                            streaming_placeholder.empty()
                            streaming_content = ""
                            streaming_placeholder = None

                        if 'DEBUG' in globals() and DEBUG:
                            st.caption("🔎 final AI message content")
                            st.code(msg.content[:4000])

                        write_any(msg.content)


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
                            tool_result = await anext(messages_agen, None)
                            if tool_result is None:
                                # Stream ended unexpectedly; tidy up and stop gracefully
                                status.update(state="complete")
                                break

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
                            if DEBUG:
                                with status:
                                    st.caption("🔎 tool_result.content")
                                    st.code(str(tool_result.content)[:4000])
                            # Single unified writer handles plot vs text
                            write_any(tool_result.content, in_status=status)
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
            feedback_kwargs = {"comment": "In-line human feedback"}
            response = await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs=feedback_kwargs,
                trace_id=st.session_state.get("trace_id")
                or st.session_state.get("langfuse_trace_id"),
                session_id=st.session_state.get("thread_id")
                or st.session_state.get("langfuse_session_id"),
                langfuse_run_id=st.session_state.get("langfuse_run_id"),
            )
            if response.langfuse_trace_id:
                st.session_state["langfuse_trace_id"] = response.langfuse_trace_id
            if response.langfuse_run_id:
                st.session_state["langfuse_run_id"] = response.langfuse_run_id
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        emit_frontend_event(
            "frontend.feedback_submitted",
            {
                "score": normalized_score,
                "run_id": latest_run_id,
            },
        )
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_agent_msgs(messages_agen, call_results, is_new):
    """
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.
    """
    nested_popovers = {}
    status = None
    # looking for the Success tool call message
    first_msg = await anext(messages_agen, None)
    if first_msg is None:
        if status:
            status.update(state="complete")
        return
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
        sub_msg = await anext(messages_agen, None)
        if sub_msg is None:
            if status:
                status.update(state="complete")
            return
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
        if status and sub_msg.content:
            # Debug (optional)
            if 'DEBUG' in globals() and DEBUG:
                with status:
                    st.caption("🔎 handle_agent_msgs sub_msg.content")
                    st.code(str(sub_msg.content)[:4000])

            # Route via universal writer
            write_any(sub_msg.content, in_status=status)

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
