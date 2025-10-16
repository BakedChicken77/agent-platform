
import json
import time
from collections.abc import Mapping
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.types import StreamWriter

from agents.utils import CustomData
from core.langgraph import emit_runtime_event, runtime_metadata
from schema.task_data import TaskData

DATA_PREVIEW_LIMIT = 400


def _summarize_payload(payload: Mapping[str, Any] | None) -> tuple[str | None, int]:
    if not payload:
        return None, 0
    try:
        text = json.dumps(dict(payload), default=str)
    except Exception:
        text = repr(dict(payload))
    if len(text) > DATA_PREVIEW_LIMIT:
        preview = text[: DATA_PREVIEW_LIMIT - 3] + "..."
    else:
        preview = text
    return preview, len(text)


class Task:
    def __init__(
        self,
        task_name: str,
        writer: StreamWriter | None = None,
        *,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = task_name
        self.id = str(uuid4())
        self.state: Literal["new", "running", "complete"] = "new"
        self.result: Literal["success", "error"] | None = None
        self.writer = writer
        self._runtime = dict(runtime) if runtime else {}
        self._started_at: float | None = None

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

    def _emit_event(self, action: str, data: Mapping[str, Any] | None = None) -> None:
        if not self._runtime:
            return
        preview, data_size = _summarize_payload(data)
        duration_ms: float | None = None
        if action == "finish" and self._started_at is not None:
            duration_ms = (time.perf_counter() - self._started_at) * 1000.0
        metadata = runtime_metadata(
            self._runtime,
            task_name=self.name,
            task_id=self.id,
            task_state=self.state,
            task_result=self.result,
            action=action,
            data_preview=preview,
            data_size=data_size or None,
            duration_ms=round(duration_ms, 3) if duration_ms is not None else None,
        )
        emit_runtime_event(f"agent.task.{self.name}", self._runtime, metadata=metadata)

    def start(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        self.state = "new"
        self._started_at = time.perf_counter()
        task_message = self._generate_and_dispatch_message(writer, data)
        self._emit_event("start", data)
        return task_message

    def write_data(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        if self.state == "complete":
            raise ValueError("Only incomplete tasks can output data.")
        self.state = "running"
        task_message = self._generate_and_dispatch_message(writer, data)
        self._emit_event("update", data)
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
        self._emit_event("finish", data)
        return task_message

