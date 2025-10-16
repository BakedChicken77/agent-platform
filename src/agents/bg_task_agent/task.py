import asyncio
import logging
from typing import Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.types import StreamWriter

from agents.utils import CustomData
from core.observability import (
    log_langfuse_event,
    sanitize_for_observability,
    snapshot_langfuse_context,
)
from schema.task_data import TaskData

logger = logging.getLogger(__name__)


class Task:
    DEFAULT_HEARTBEAT_SECONDS = 30.0

    def __init__(
        self,
        task_name: str,
        writer: StreamWriter | None = None,
        *,
        heartbeat_interval: float | None = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        self.name = task_name
        self.id = str(uuid4())
        self.state: Literal["new", "running", "complete"] = "new"
        self.result: Literal["success", "error"] | None = None
        self.writer = writer
        self._heartbeat_interval = heartbeat_interval if heartbeat_interval and heartbeat_interval > 0 else None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_stop: asyncio.Event | None = None
        self._langfuse_context = snapshot_langfuse_context()

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

    def _log_transition(self, phase: str, data: dict) -> None:
        metadata = {
            "task_id": self.id,
            "task_name": self.name,
            "phase": phase,
            "state": self.state,
        }
        if self.result:
            metadata["result"] = self.result
        if data:
            metadata["data_preview"] = sanitize_for_observability(data)
        log_langfuse_event(
            name=f"task.{phase}",
            input_payload=data,
            metadata=metadata,
            context=self._langfuse_context,
        )

    def _heartbeat_done(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Task heartbeat ended with error: %s", exc)

    async def _heartbeat_loop(self) -> None:
        if not self._heartbeat_stop or not self._heartbeat_interval:
            return
        try:
            while True:
                try:
                    await asyncio.wait_for(self._heartbeat_stop.wait(), timeout=self._heartbeat_interval)
                    break
                except asyncio.TimeoutError:
                    log_langfuse_event(
                        name="task.heartbeat",
                        metadata={
                            "task_id": self.id,
                            "task_name": self.name,
                            "state": self.state,
                            "result": self.result,
                        },
                        context=self._langfuse_context,
                    )
        except asyncio.CancelledError:
            raise

    def _start_heartbeat(self) -> None:
        if self._heartbeat_interval is None or self._heartbeat_task:
            return
        if self._langfuse_context is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._heartbeat_stop = asyncio.Event()
        self._heartbeat_task = loop.create_task(self._heartbeat_loop())
        self._heartbeat_task.add_done_callback(self._heartbeat_done)

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_stop and not self._heartbeat_stop.is_set():
            self._heartbeat_stop.set()
        self._heartbeat_stop = None
        self._heartbeat_task = None

    def start(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        self.state = "new"
        task_message = self._generate_and_dispatch_message(writer, data)
        self._log_transition("start", data)
        self._start_heartbeat()
        return task_message

    def write_data(self, writer: StreamWriter | None = None, data: dict = {}) -> BaseMessage:
        if self.state == "complete":
            raise ValueError("Only incomplete tasks can output data.")
        self.state = "running"
        task_message = self._generate_and_dispatch_message(writer, data)
        self._log_transition("update", data)
        self._start_heartbeat()
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
        self._log_transition("finish", data)
        self._stop_heartbeat()
        return task_message

