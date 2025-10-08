import asyncio
import base64
import builtins
import io
import math
import os
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import matplotlib
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
from schema.files import FileMeta
from service.catalog import list_metadata

matplotlib.use("Agg")  # force headless backend


@dataclass(frozen=True)
class _UploadEntry:
    """Record describing a user-uploaded file that is safe to expose."""

    id: str
    name: str
    path: Path
    mime: str
    size: int


@dataclass
class _UploadContext:
    """Holds lookup tables for the currently executing python_repl call."""

    entries_by_id: dict[str, _UploadEntry]
    entries_by_name: dict[str, list[_UploadEntry]]
    entries_by_path: dict[Path, _UploadEntry]

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "id": entry.id,
                "name": entry.name,
                "mime": entry.mime,
                "size": entry.size,
            }
            for entry in self.entries_by_id.values()
        ]


class UploadPath:
    """Restricted, read-only handle to an uploaded file."""

    __slots__ = ("_entry",)

    def __init__(self, entry: _UploadEntry):
        self._entry = entry

    @property
    def id(self) -> str:
        return self._entry.id

    @property
    def name(self) -> str:
        return self._entry.name

    @property
    def mime(self) -> str:
        return self._entry.mime

    @property
    def size(self) -> int:
        return self._entry.size

    def open(self, mode: str = "rb", *args: Any, **kwargs: Any):
        return _safe_open(self, mode=mode, *args, **kwargs)

    def read_bytes(self) -> bytes:
        with self.open("rb") as fp:
            return fp.read()

    def read_text(self, encoding: str = "utf-8") -> str:
        with self.open("r", encoding=encoding) as fp:
            return fp.read()

    def as_posix(self) -> str:
        return self._entry.path.as_posix()

    def __fspath__(self) -> str:
        return str(self._entry.path)

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return str(self._entry.path)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"UploadPath(id={self._entry.id!r}, name={self._entry.name!r})"


_upload_context: ContextVar[_UploadContext | None] = ContextVar(
    "upload_context", default=None
)
_UPLOAD_ROOT = Path(settings.UPLOAD_DIR).resolve()


def _ensure_within_root(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=False)
    except FileNotFoundError:
        resolved = path.resolve()

    if hasattr(resolved, "is_relative_to"):
        if not resolved.is_relative_to(_UPLOAD_ROOT):  # type: ignore[attr-defined]
            raise PermissionError("Access outside upload directory is not allowed")
    else:  # pragma: no cover - Python >=3.9 always has is_relative_to
        try:
            resolved.relative_to(_UPLOAD_ROOT)
        except ValueError as exc:
            raise PermissionError("Access outside upload directory is not allowed") from exc
    return resolved


def _get_context() -> _UploadContext:
    ctx = _upload_context.get()
    if ctx is None:
        raise RuntimeError(
            "File helpers are only available inside python_repl execution contexts."
        )
    return ctx


def _resolve_entry(target: Any) -> _UploadEntry:
    ctx = _get_context()
    if isinstance(target, UploadPath):
        return ctx.entries_by_id[target.id]

    if isinstance(target, str):
        if target in ctx.entries_by_id:
            return ctx.entries_by_id[target]
        matches = ctx.entries_by_name.get(target)
        if matches:
            if len(matches) > 1:
                raise ValueError(
                    "Multiple uploads share this name. Refer to the file by its ID instead."
                )
            return matches[0]
        potential_path = Path(target)
    elif isinstance(target, os.PathLike):
        potential_path = Path(target)
    else:
        raise TypeError("Unsupported target type for file resolution")

    resolved = _ensure_within_root(potential_path if potential_path.is_absolute() else _UPLOAD_ROOT / potential_path)
    entry = ctx.entries_by_path.get(resolved)
    if entry is None:
        raise FileNotFoundError("Requested file is not in the current user's uploads")
    return entry


def _safe_open(target: Any, mode: str = "r", *args: Any, **kwargs: Any):
    if set(mode) & {"w", "a", "+"}:
        raise PermissionError("File writes are disabled inside python_repl")
    entry = _resolve_entry(target)
    return builtins.open(entry.path, mode, *args, **kwargs)


def list_uploads() -> list[dict[str, Any]]:
    """Return metadata for files available to the current session."""

    return _get_context().summary()


def resolve_upload(target: Any) -> UploadPath:
    """Resolve an upload identifier to a read-only UploadPath handle."""

    return UploadPath(_resolve_entry(target))


def load_upload_bytes(target: Any) -> bytes:
    """Return the raw bytes of an uploaded file."""

    return resolve_upload(target).read_bytes()


def load_upload_text(target: Any, encoding: str = "utf-8") -> str:
    """Return the text content of an uploaded file."""

    return resolve_upload(target).read_text(encoding=encoding)


def _coerce_to_safe_pathlike(target: Any) -> Any:
    if isinstance(target, str | os.PathLike[str] | UploadPath):
        return _resolve_entry(target).path
    return target


_pd_read_excel = pd.read_excel
_pd_read_csv = getattr(pd, "read_csv", None)


def _safe_read_excel(io: Any, *args: Any, **kwargs: Any):
    return _pd_read_excel(_coerce_to_safe_pathlike(io), *args, **kwargs)


def _safe_read_csv(io: Any, *args: Any, **kwargs: Any):
    if _pd_read_csv is None:  # pragma: no cover - pandas always exposes read_csv
        raise AttributeError("pandas.read_csv is unavailable")
    return _pd_read_csv(_coerce_to_safe_pathlike(io), *args, **kwargs)


pd.read_excel = _safe_read_excel  # type: ignore[assignment]
if _pd_read_csv is not None:
    pd.read_csv = _safe_read_csv  # type: ignore[assignment]


# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()

# Initialize LLM
model = get_model(settings.DEFAULT_MODEL)

safe_globals = {
    "__builtins__": {"print": print},
    "math": math,
    "numpy": np,
    "np": np,
    "plotly": plotly,
    "pio": pio,
    "plt": plt,
    "io": io,
    "base64": base64,
    "open": _safe_open,
    "Path": resolve_upload,
    "pd": pd,
    "list_uploads": list_uploads,
    "resolve_upload": resolve_upload,
    "load_upload_bytes": load_upload_bytes,
    "load_upload_text": load_upload_text,
}


def _state_to_ids(state: MessagesState) -> tuple[str | None, str | None]:
    cfg = state.get("configurable", {}) if isinstance(state, dict) else {}
    return cfg.get("user_id"), cfg.get("thread_id")


async def _load_context(state: MessagesState) -> _UploadContext:
    user_id, thread_id = _state_to_ids(state)
    store = getattr(coding_agent, "store", None)
    if not user_id or store is None:
        return _UploadContext({}, {}, {})

    try:
        records: Iterable[FileMeta] = await list_metadata(store, user_id=user_id, thread_id=thread_id)
    except Exception:
        return _UploadContext({}, {}, {})

    entries_by_id: dict[str, _UploadEntry] = {}
    entries_by_name: dict[str, list[_UploadEntry]] = {}
    entries_by_path: dict[Path, _UploadEntry] = {}

    for meta in records:
        path = _ensure_within_root(Path(meta.path))
        if not path.exists():
            continue
        entry = _UploadEntry(
            id=meta.id,
            name=meta.original_name,
            path=path,
            mime=meta.mime,
            size=meta.size,
        )
        entries_by_id[entry.id] = entry
        entries_by_name.setdefault(entry.name, []).append(entry)
        entries_by_path[path] = entry

    return _UploadContext(entries_by_id, entries_by_name, entries_by_path)


@tool
async def python_repl(
    code: str, state: Annotated[MessagesState, InjectedState]
) -> str:
    """Execute Python code and return its stdout or an error message."""

    repl = PythonREPL(globals=safe_globals)
    context = await _load_context(state)
    token = _upload_context.set(context)
    try:
        return await asyncio.to_thread(repl.run, code)
    except Exception as e:  # pragma: no cover - passthrough for runtime errors
        return f"Execution failed: {e}"
    finally:
        _upload_context.reset(token)


prompt_coding_agent = """
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
- To inspect uploads use `list_uploads()`; resolve a file with `Path(...)` or `resolve_upload(...)`.
- All files are read-only. Use `open`, `load_upload_bytes`, or pandas helpers (e.g. `pd.read_excel(Path("<file-id>"))`).
- Do NOT call plt.show(). Do NOT open GUI windows. Do NOT write files to disk.
- Only print one payload line (PLOTLY_JSON:... or DATA_URI:...).
"""

 
# Create the coding expert agent
coding_agent = create_react_agent(
    model=model,
    tools=[python_repl],
    name="coding_expert",
    prompt=prompt_coding_agent
).with_config(tags=["skip_stream"])
