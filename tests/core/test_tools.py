from contextlib import AbstractContextManager

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agents.bg_task_agent.task import Task
from core.tools import instrument_langfuse_tool


class _NoOpContext(AbstractContextManager):
    def __enter__(self):  # pragma: no cover - trivial
        return None

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        return False


class _DummySpan:
    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.ended: dict[str, object] | None = None

    def start_as_current_span(self):  # pragma: no cover - simple passthrough
        return _NoOpContext()

    def update(self, **payload):
        self.updates.append(payload)

    def end(self, level: str | None = None, status_message: str | None = None):
        self.ended = {"level": level, "status_message": status_message}


class _DummyClient:
    def __init__(self) -> None:
        self.span_calls: list[dict] = []
        self.events: list[dict] = []
        self._spans: list[_DummySpan] = []

    def span(self, **kwargs):
        self.span_calls.append(kwargs)
        span = _DummySpan()
        self._spans.append(span)
        return span

    def event(self, **kwargs):
        self.events.append(kwargs)


def _runtime_config() -> RunnableConfig:
    return RunnableConfig(
        configurable={
            "langfuse": {
                "trace_id": "trace",
                "session_id": "session",
                "agent_name": "agent",
                "run_id": "run",
                "thread_id": "thread",
                "user_id": "user",
            },
            "trace_id": "trace",
            "session_id": "session",
            "user_id": "user",
            "thread_id": "thread",
        }
    )


def test_instrument_langfuse_tool_emits_events(monkeypatch):
    client = _DummyClient()
    monkeypatch.setattr("core.langgraph.get_langfuse_client", lambda: client)

    @tool
    def sample(value: str) -> str:
        return value.upper()

    sample_tool = instrument_langfuse_tool(sample, name="sample_tool")

    result = sample_tool.invoke("hello", config=_runtime_config())

    assert result == "HELLO"
    assert client.span_calls[0]["name"] == "agent.tool.sample_tool"
    assert client.events and client.events[0]["name"] == "agent.tool.sample_tool"
    metadata = client.events[0]["metadata"]
    assert metadata["tool"] == "sample_tool"
    assert metadata["status"] == "success"
    assert "input_preview" in metadata
    assert client._spans[0].updates  # span updates captured output metadata


def test_instrument_langfuse_tool_without_client(monkeypatch):
    monkeypatch.setattr("core.langgraph.get_langfuse_client", lambda: None)

    @tool
    def echo(value: str) -> str:
        return value

    echo_tool = instrument_langfuse_tool(echo, name="echo")
    assert echo_tool.invoke("ping", config=_runtime_config()) == "ping"


def test_task_emits_langfuse_events(monkeypatch):
    events: list[tuple[str, dict[str, str | None] | None]] = []

    def _capture_event(name, runtime, metadata=None):
        events.append((name, metadata))

    monkeypatch.setattr("agents.bg_task_agent.task.emit_runtime_event", _capture_event)

    runtime = {"trace_id": "trace", "session_id": "session", "user_id": "user"}
    task = Task("demo", runtime=runtime)

    task.start(data={"foo": "bar"})
    task.write_data(data={"progress": 50})
    task.finish("success", data={"result": "ok"})

    assert [name for name, _ in events] == [
        "agent.task.demo",
        "agent.task.demo",
        "agent.task.demo",
    ]
    assert events[0][1]["action"] == "start"
    assert events[-1][1]["task_result"] == "success"
    assert "data_preview" in events[0][1]

    events.clear()
    idle_task = Task("idle")
    idle_task.start()
    assert events == []
