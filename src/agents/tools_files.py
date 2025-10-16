from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

from langchain_core.tools import tool

from core import settings
from core.observability import instrument_tool
from schema.files import FileMeta, ListFilesResponse
from service.catalog import list_metadata, get_metadata
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState


def _ctx_ids(state: MessagesState) -> tuple[str, str | None]:
    cfg = state.get("configurable", {}) if isinstance(state, dict) else {}
    user_id = cfg.get("user_id")
    thread_id = cfg.get("thread_id")
    if not user_id:
        raise ValueError("Missing configurable.user_id in runtime config")
    return user_id, thread_id


@tool
@instrument_tool(name="ListUserFiles")
async def ListUserFiles(
    state: Annotated[MessagesState, InjectedState],
) -> list[dict]:
    """List metadata for the current user's files (scoped by optional thread_id)."""
    from agents import DEFAULT_AGENT, get_agent
    agent = get_agent(DEFAULT_AGENT)
    store = getattr(agent, "store", None)
    if store is None:
        return []
    user_id, thread_id = _ctx_ids(state)
    items = await list_metadata(store, user_id=user_id, thread_id=thread_id)
    return [m.model_dump() for m in items]

@tool
@instrument_tool(name="ReadUserFile")
async def ReadUserFile(
    file_id: Annotated[str, "ID returned by list files"],
    state: Annotated[MessagesState, InjectedState],
    as_text: Annotated[bool, "If true, return UTF-8 text, else bytes repr"] = True,
) -> str:
    """Read a user's file by ID; returns text (utf-8) or bytes (repr)."""
    from agents import DEFAULT_AGENT, get_agent
    agent = get_agent(DEFAULT_AGENT)
    store = getattr(agent, "store", None)
    if store is None:
        return "[error] store not initialized"

    user_id, thread_id = _ctx_ids(state)
    meta = await get_metadata(store, user_id=user_id, file_id=file_id, thread_id=thread_id)
    if not meta:
        return f"[error] file not found: {file_id}"
    p = Path(meta.path)
    if not p.exists():
        return f"[error] file missing on disk: {file_id}"
    data = p.read_bytes()
    if as_text:
        return data.decode("utf-8", errors="replace")
    return repr(data)
