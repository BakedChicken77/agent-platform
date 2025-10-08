# coding_agent.py

import base64
import inspect
import io
import math
from pathlib import Path
from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")  # force headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from core import get_model, settings
from langgraph.graph import MessagesState
from langgraph.prebuilt import InjectedState, create_react_agent
from service import catalog_postgres

# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()

# Initialize LLM
model = get_model(settings.DEFAULT_MODEL)

# safe_globals = {
#     "__builtins__": {"print": print},
#     "math": math,
#     "numpy": np,
#     "np": np,  # <-- optional, for convenience
#     "plotly": plotly,
# }
safe_globals: dict[str, Any] = {
    "__builtins__": {"print": print},
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


def _extract_runtime_ids(state: MessagesState | dict[str, Any]) -> tuple[str | None, str | None]:
    config: dict[str, Any] = {}
    if isinstance(state, dict):
        config = state.get("configurable", {}) or {}
    elif hasattr(state, "get"):
        try:
            config = state.get("configurable", {})  # type: ignore[arg-type]
        except Exception:
            config = {}
    user_id = config.get("user_id")
    thread_id = config.get("thread_id")
    return user_id, thread_id


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
    state: Annotated[MessagesState, InjectedState] | None = None,
) -> str:
    """
    Execute Python code and return its stdout or an error message.
    """
    user_id: str | None = None
    thread_id: str | None = None
    if state is not None:
        user_id, thread_id = _extract_runtime_ids(state)
    repl_globals = await _prepare_repl_globals(user_id, thread_id)
    repl = PythonREPL(globals=repl_globals)
    try:
        return repl.run(code)
    except Exception as e:
        return f"Execution failed: {e}"



# prompt_coding_agent = """\
# You are a senior Python coding assistant specialized in data-science and exploratory data analysis.
# Follow these rules:
# - Generate correct, runnable Python code.
# - Always wrap your code in a single python_repl tool call.
# - If the user asks for explanation, design advice, or non-code discussion, respond directly in plain text.
# - The runtime already includes: `math`, `numpy` as `np` and `plotly`. Do NOT use `import` statements — \
# they are unnecessary and disabled. Simply write code assuming those modules are already available.
# - DO NOT attempt to display plots via popup window. Instead, create .png's for all plots and save them in the workspace.
# """

prompt_coding_agent = """\
You are a senior Python coding assistant specialized in data-science and plotting.

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

 
# Create the coding expert agent
coding_agent = create_react_agent(
    model=model,
    tools=[python_repl],
    name="coding_expert",
    prompt=prompt_coding_agent
).with_config(tags=["skip_stream"])
