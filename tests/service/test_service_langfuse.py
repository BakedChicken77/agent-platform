from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import UUID

import pytest

from schema import StreamInput
from service.service import (
    LangfuseTelemetry,
    _end_span,
    _handle_input,
    _start_langfuse_span,
)


class _DummySpan:
    def __init__(self) -> None:
        self.started = False
        self.ended: list[dict[str | None, str | None]] = []

    @contextmanager
    def start_as_current_span(self):
        self.started = True
        yield

    def end(self, level: str | None = None, status_message: str | None = None):
        self.ended.append({"level": level, "status_message": status_message})


class _DummyClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def span(self, **kwargs):
        self.calls.append(kwargs)
        return _DummySpan()


def _telemetry() -> LangfuseTelemetry:
    return LangfuseTelemetry(
        trace_id="trace-123",
        session_id="session-456",
        user_id="user-abc",
        metadata={"agent_id": "demo", "thread_id": "thread-1", "user_id": "user-abc"},
    )


def test_start_langfuse_span_uses_client(monkeypatch):
    client = _DummyClient()
    monkeypatch.setattr("service.service.get_langfuse_client", lambda: client)
    telemetry = _telemetry()

    span, context = _start_langfuse_span("agent.test", telemetry)

    assert isinstance(span, _DummySpan)
    assert client.calls[0]["name"] == "agent.test"
    assert client.calls[0]["trace_id"] == telemetry.trace_id
    assert client.calls[0]["session_id"] == telemetry.session_id
    assert client.calls[0]["user_id"] == telemetry.user_id

    with context:
        pass
    assert span.started is True


def test_end_span_marks_errors():
    span = _DummySpan()
    _end_span(span)
    assert span.ended[-1] == {"level": None, "status_message": None}

    error = ValueError("boom")
    _end_span(span, error=error)
    assert span.ended[-1]["level"] == "ERROR"
    assert span.ended[-1]["status_message"] == "ValueError: boom"


@pytest.mark.asyncio
async def test_handle_input_attaches_langfuse_callbacks(monkeypatch):
    handler_calls: dict[str, list[str]] = {"trace": [], "session": [], "user": []}

    class DummyHandler:
        def set_trace_id(self, value: str) -> None:
            handler_calls["trace"].append(value)

        def set_session_id(self, value: str) -> None:
            handler_calls["session"].append(value)

        def set_user_id(self, value: str) -> None:
            handler_calls["user"].append(value)

    class DummyTrace:
        def __init__(self) -> None:
            self.handler = DummyHandler()

        def get_langchain_handler(self) -> DummyHandler:
            return self.handler

    class DummyClient:
        def __init__(self) -> None:
            self.trace_calls: list[dict[str, object]] = []

        def trace(self, **kwargs):
            self.trace_calls.append(kwargs)
            return DummyTrace()

    dummy_client = DummyClient()
    monkeypatch.setattr("service.service.get_langfuse_client", lambda: dummy_client)
    monkeypatch.setattr("service.service.get_langfuse_handler", lambda: None)
    monkeypatch.setattr("service.service.settings", SimpleNamespace(LANGFUSE_TRACING=True))

    class DummyAgent:
        async def aget_state(self, *, config):
            return SimpleNamespace(tasks=[], values={"messages": []})

    user_input = StreamInput(message="hello", stream_tokens=False)

    kwargs, run_id, telemetry = await _handle_input(
        user_input,
        DummyAgent(),
        claims={"oid": "user-123"},
        agent_id="demo-agent",
    )

    assert isinstance(run_id, UUID)
    config = kwargs["config"]
    callbacks = config.get("callbacks", []) if isinstance(config, dict) else []
    assert callbacks and isinstance(callbacks[0], DummyHandler)
    assert telemetry.handlers and telemetry.handlers[0] is callbacks[0]
    assert handler_calls["trace"] == [telemetry.trace_id]
    assert handler_calls["session"] == [telemetry.session_id]
    assert handler_calls["user"] == [telemetry.user_id]

