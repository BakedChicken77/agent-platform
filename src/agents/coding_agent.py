# src/agents/coding_agent.py

import base64
import inspect
import io
import math
from pathlib import Path
from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")  # force headless backend

import builtins

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict

from agents.instrumentation import configure_agent, with_langfuse_span
from core import get_model, settings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import InjectedState, create_react_agent
from service import catalog_postgres

# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()

# Initialize LLM
model = get_model(settings.DEFAULT_MODEL)

safe_globals: dict[str, Any] = {
    "__builtins__": builtins, #{"print": print},
    "math": math,
    "numpy": np,
    "np": np,
    "plotly": plotly,
    "pio": pio,
    "plt": plt,
    "io": io,
    "base64": base64,
    "pd": pd,
    "pandas": pd,
}


class CodingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: RemainingSteps  # required by the built-in agent
    runtime_ids: dict[str, Any]      # <-- keep your injected IDs



def _extract_runtime_ids(state: MessagesState) -> tuple[str | None, str | None]:
    """
    Read IDs injected by pre_model_hook under `runtime_ids`.
    """
    cfg = state.get("runtime_ids", {}) or {}
    return cfg.get("user_id"), cfg.get("thread_id")


async def _load_user_uploads(user_id: str | None, thread_id: str | None) -> list[Any]:
    if not user_id:
        return []
    try:
        records = catalog_postgres.list_metadata(user_id=user_id, thread_id=thread_id)
        if inspect.isawaitable(records):
            return await records  # type: ignore[return-value]
        return records
    except Exception:
        return []


async def _prepare_repl_globals(user_id: str | None, thread_id: str | None) -> dict[str, Any]:
    context_globals: dict[str, Any] = dict(safe_globals)
    file_map: dict[str, str] = {}
    metadata: list[dict[str, Any]] = []

    for meta in await _load_user_uploads(user_id, thread_id):
        path_value = getattr(meta, "path", None)
        if not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
        path_str = str(path.resolve())
        file_map.setdefault(meta.original_name, path_str)
        file_map.setdefault(meta.id, path_str)
        metadata.append(
            {
                "id": meta.id,
                "original_name": meta.original_name,
                "mime": meta.mime,
                "size": meta.size,
                "created_at": meta.created_at,
                "indexed": meta.indexed,
                "path": path_str,
            }
        )

    metadata_snapshot = [item.copy() for item in metadata]

    def list_uploaded_files() -> list[dict[str, Any]]:
        return [item.copy() for item in metadata_snapshot]

    def load_uploaded_file(name: str, *, as_bytes: bool = True):
        path_str = file_map.get(name)
        if not path_str:
            raise FileNotFoundError(f"Unknown uploaded file: {name}")
        data = Path(path_str).read_bytes()
        if as_bytes:
            buffer = io.BytesIO(data)
            buffer.seek(0)
            return buffer
        return data.decode("utf-8", errors="replace")

    context_globals["uploaded_file_paths"] = dict(file_map)
    context_globals["uploaded_file_metadata"] = metadata_snapshot
    context_globals["list_uploaded_files"] = list_uploaded_files
    context_globals["load_uploaded_file"] = load_uploaded_file

    return context_globals


@tool
async def python_repl(
    code: str,
    state: Annotated[CodingState, InjectedState] | None = None,
) -> str:
    """
    Execute Python code and return its stdout or an error message.
    """
    user_id: str | None = None
    thread_id: str | None = None

    if state is not None:
        user_id, thread_id = _extract_runtime_ids(state)

    repl_globals = await _prepare_repl_globals(user_id, thread_id)
    repl = PythonREPL(_globals=repl_globals, _locals=repl_globals)

    try:
        return repl.run(code)
    except Exception as e:
        return f"Execution failed: {e}"


prompt_coding_agent = """\
You are a senior Python coding assistant specialized in data-science and plotting.

Important:
- The python_repl tool only returns what is printed to stdout.
- Always use print(...) to display any value, object, list, or DataFrame.
- For example: use `print(list_uploaded_files())` instead of just `list_uploaded_files()`.
- Expressions without print() will return no visible output.

Rules:
- Wrap ALL runnable code in a single python_repl tool call.
- Prefer Plotly. If you create a Plotly figure `fig`, finish by:
    print("PLOTLY_JSON:" + fig.to_json())
- If you use Matplotlib (via preloaded `plt`), finish by writing the figure to an in-memory PNG:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    import base64  # already available
    print("DATA_URI:data:image/png;base64," + base64.b64encode(buf.read()).decode())
- Do NOT call plt.show(). Do NOT open GUI windows. Do NOT write files to disk.
- Only print one payload line (PLOTLY_JSON:... or DATA_URI:...).
- Uploaded files for this chat are available:
    • `uploaded_file_paths` maps original filenames and file IDs to absolute paths.
    • `uploaded_file_metadata` contains metadata dictionaries (with "path").
    • `list_uploaded_files()` returns metadata copies.
    • `load_uploaded_file(name, as_bytes=True)` returns a BytesIO (default) or UTF-8 text when `as_bytes=False`.
- Use the preloaded `pd` (pandas) module for tabular data, including Excel files.
"""


def _pre_model_inject_runtime_ids(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    cfg = config.get("configurable", {}) or {}
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        messages = list(messages)
    return {
        "messages": messages,
        "runtime_ids": {
            "user_id": cfg.get("user_id"),
            "thread_id": cfg.get("thread_id"),
        },
    }


# Create the coding expert agent with a pre_model_hook to inject runtime IDs
coding_agent = create_react_agent(
    model=model,
    tools=[python_repl],
    prompt=prompt_coding_agent,
    pre_model_hook=_pre_model_inject_runtime_ids,
    state_schema=CodingState,          # <-- IMPORTANT
    name="coding_expert",
).with_config(tags=["skip_stream"])


builder = StateGraph(MessagesState)


@with_langfuse_span("react_agent")
async def _react_agent_node(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    return await coding_agent.ainvoke(state, config)


builder.add_node("react_agent", _react_agent_node)
builder.add_edge(START, "react_agent")
builder.add_edge("react_agent", END)

# 3) Re-export under the same symbol & name
coding_agent2 = builder.compile()
coding_agent2.name = "coding_expert"
coding_agent2 = configure_agent(
    coding_agent2,
    agent_id=coding_agent2.name,
    agent_kind="state_graph",
    description="A Python Coding Agent",
)
