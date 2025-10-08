from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from agents import coding_agent


class _DummyStore:
    def __init__(self, rows: Iterable[tuple[str, dict]]):
        self._rows = list(rows)

    async def ascan(self, collection: str, namespace: str):  # pragma: no cover - interface stub
        return list(self._rows)


@pytest.mark.asyncio
async def test_python_repl_can_read_uploaded_excel(tmp_path: Path, monkeypatch):
    user_id = "user-123"
    thread_id = "thread-456"
    file_id = "file-abc"

    upload_root = tmp_path / "uploads"
    file_path = upload_root / user_id / thread_id / f"{file_id}.xlsx"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df_source = pd.DataFrame({"value": [42]})
    df_source.to_excel(file_path, index=False)

    metadata = {
        "id": file_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "tenant_id": None,
        "original_name": "sample.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "size": file_path.stat().st_size,
        "sha256": "0" * 64,
        "path": str(file_path),
        "created_at": int(time.time()),
        "indexed": False,
    }

    monkeypatch.setattr(coding_agent, "_UPLOAD_ROOT", upload_root.resolve())
    original_store = getattr(coding_agent.coding_agent, "store", None)
    coding_agent.coding_agent.store = _DummyStore([(file_id, metadata)])

    try:
        code = """
files = list_uploads()
target = files[0]["id"]
df = pd.read_excel(Path(target))
print(df.to_dict())
"""

        result = await coding_agent.python_repl.ainvoke(
            {"code": code},
            config={"configurable": {"user_id": user_id, "thread_id": thread_id}},
        )
    finally:
        coding_agent.coding_agent.store = original_store

    assert "{'value': {0: 42}}" in result
