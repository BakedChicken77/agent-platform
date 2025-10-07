from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from langgraph.store.base import BaseStore

from schema.files import FileMeta


# We store records under collection="files", namespace=user_id (optionally "user_id:thread_id")
COLLECTION = "files"


def _ns(user_id: str, thread_id: str | None) -> str:
    return f"{user_id}:{thread_id}" if thread_id else user_id


async def save_metadata(store: BaseStore, meta: FileMeta) -> None:
    data = meta.model_dump()
    await store.aput(COLLECTION, _ns(meta.user_id, meta.thread_id), meta.id, data)


async def get_metadata(store: BaseStore, user_id: str, file_id: str, thread_id: str | None = None) -> FileMeta | None:
    rec = await store.aget(COLLECTION, _ns(user_id, thread_id), file_id)
    return FileMeta.model_validate(rec) if rec else None


async def list_metadata(store: BaseStore, user_id: str, thread_id: str | None = None) -> list[FileMeta]:
    rows: Iterable[tuple[str, Any]] = await store.ascan(COLLECTION, _ns(user_id, thread_id))
    out: list[FileMeta] = []
    for _, v in rows:
        try:
            out.append(FileMeta.model_validate(v))
        except Exception:
            continue
    # newest first
    out.sort(key=lambda m: m.created_at, reverse=True)
    return out


async def delete_metadata(store: BaseStore, user_id: str, file_id: str, thread_id: str | None = None) -> None:
    await store.adelete(COLLECTION, _ns(user_id, thread_id), file_id)
