from contextlib import contextmanager

import pytest
from langchain_core.runnables import RunnableConfig

from core.langgraph import (
    LANGFUSE_STATE_KEY,
    instrument_langgraph_node,
    persist_langfuse_state,
)


class DummySpan:
    def __init__(self) -> None:
        self.started = False
        self.ended = False
        self.status_message: str | None = None

    @contextmanager
    def start_as_current_span(self):
        self.started = True
        yield

    def end(self, level: str | None = None, status_message: str | None = None):
        self.ended = True
        self.status_message = status_message


class DummyClient:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def span(self, **kwargs):
        self.last_kwargs = kwargs
        return DummySpan()


@pytest.mark.asyncio
async def test_instrument_langgraph_node_creates_span(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr("core.langgraph.get_langfuse_client", lambda: dummy)

    async def node(state, *, config: RunnableConfig):
        state.setdefault("messages", []).append("ok")
        return state

    wrapped = instrument_langgraph_node(node, "test.node")
    config = RunnableConfig(
        configurable={
            "langfuse": {
                "trace_id": "trace-123",
                "session_id": "session-abc",
                "agent_name": "demo",
                "run_id": "run-456",
                "user_id": "user-1",
                "thread_id": "thread-2",
            }
        }
    )

    state = {"messages": []}
    result = await wrapped(state, config=config)

    assert result["messages"] == ["ok"]
    assert dummy.last_kwargs is not None
    assert dummy.last_kwargs["trace_id"] == "trace-123"
    assert dummy.last_kwargs["session_id"] == "session-abc"
    assert dummy.last_kwargs["metadata"]["agent_name"] == "demo"


def test_persist_langfuse_state_merges_sources():
    config = RunnableConfig(
        configurable={
            "langfuse": {"trace_id": "abc"},
            "session_id": "session-1",
            "agent_name": "agent-x",
            "run_id": "run-99",
        }
    )
    previous = {LANGFUSE_STATE_KEY: {"run_id": "persisted", "trace_id": "should-be-overridden"}}
    payload = {"messages": []}

    result = persist_langfuse_state(payload, config=config, previous=previous)

    assert result[LANGFUSE_STATE_KEY]["trace_id"] == "abc"
    assert result[LANGFUSE_STATE_KEY]["session_id"] == "session-1"
    assert result[LANGFUSE_STATE_KEY]["run_id"] == "run-99"
    assert result[LANGFUSE_STATE_KEY]["agent_name"] == "agent-x"
