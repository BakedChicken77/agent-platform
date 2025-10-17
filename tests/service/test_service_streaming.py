
import json
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from schema import ChatMessage, StreamInput
from service.service import LangfuseTelemetry, _create_ai_message, message_generator


@pytest.mark.parametrize(
    "parts, expected",
    [
        # 1) Basic content + tool_calls
        (
            {"content": "Hello", "tool_calls": []},
            {"content": "Hello", "tool_calls": []},
        ),
        # 2) Unknown keys are ignored
        (
            {"content": "Test", "foobar": 123, "tool_calls": []},
            {"content": "Test", "tool_calls": []},
        ),
        # 3) Extra valid AIMessage params (id, type) pass through
        (
            {
                "content": "Hey",
                "id": "abc-123",
                "type": "ai",
                "tool_calls": [],
            },
            {"content": "Hey", "id": "abc-123", "type": "ai", "tool_calls": []},
        ),
    ],
)
def test_create_ai_message_filters_and_passes_through(parts, expected):
    """
    _create_ai_message should:
      - Drop unknown keys ("foobar")
      - Preserve keys that match AIMessage signature
      - Use the final value for duplicate keys in the parts dict
    """
    msg: AIMessage = _create_ai_message(parts)
    for key, val in expected.items():
        assert getattr(msg, key) == val


def test_create_ai_message_missing_required_content_raises():
    """
    AIMessage requires 'content'; if missing, _create_ai_message should
    bubble up the TypeError from the constructor.
    """
    with pytest.raises(TypeError):
        _create_ai_message({"tool_calls": []})


def test_create_ai_message_empty_dict_raises():
    """
    Completely empty parts should also fail to construct an AIMessage.
    """
    with pytest.raises(TypeError):
        _create_ai_message({})


@pytest.mark.asyncio
async def test_message_generator_includes_langfuse_metadata(monkeypatch):
    run_id = UUID("12345678-1234-5678-1234-567812345678")
    telemetry = LangfuseTelemetry(
        trace_id="trace-xyz",
        session_id="session-xyz",
        user_id="user-42",
        metadata={"agent_id": "demo", "thread_id": "thread-9", "user_id": "user-42"},
    )

    callbacks = [object()]
    config = RunnableConfig(configurable={}, callbacks=callbacks, run_id=run_id)

    class DummyAgent:
        def __init__(self) -> None:
            self.last_config = None

        async def astream(self, *, input, config, stream_mode, subgraphs):
            self.last_config = config
            yield ("custom", ChatMessage(type="ai", content="hello"))

    dummy_agent = DummyAgent()

    async def fake_handle_input(user_input, agent, claims, agent_id):
        return {"input": {}, "config": config}, run_id, telemetry

    monkeypatch.setattr("service.service.get_agent", lambda agent_id: dummy_agent)
    monkeypatch.setattr("service.service._handle_input", fake_handle_input)
    monkeypatch.setattr("service.service.langchain_to_chat_message", lambda msg: msg)
    monkeypatch.setattr("service.service._record_stream_message_span", lambda *_, **__: None)
    monkeypatch.setattr("service.service._record_stream_token_span", lambda *_, **__: None)

    user_input = StreamInput(message="hello", stream_tokens=False)

    events: list[str] = []
    async for chunk in message_generator(user_input, "demo", {}):
        events.append(chunk)
        if chunk.endswith("[DONE]\n\n"):
            break

    assert dummy_agent.last_config == config
    assert dummy_agent.last_config.get("callbacks") == callbacks

    message_event = events[0]
    assert message_event.startswith("data: ")
    payload = json.loads(message_event[len("data: ") :].strip())
    assert payload["langfuse"]["trace_id"] == telemetry.trace_id
    assert payload["langfuse"]["session_id"] == telemetry.session_id
    assert payload["content"]["custom_data"]["langfuse"]["trace_id"] == telemetry.trace_id
    assert payload["content"]["run_id"] == str(run_id)
    assert events[-1] == "data: [DONE]\n\n"

