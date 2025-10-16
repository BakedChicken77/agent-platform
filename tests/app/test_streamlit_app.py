
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest
from streamlit.testing.v1 import AppTest

from client import AgentClientError
from schema import ChatHistory, ChatMessage
from schema.models import OpenAIModelName


def test_app_simple_non_streaming(mock_agent_client):
    """Test the full app - happy path"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    WELCOME_START = "Hello! I'm an AI agent. Ask me anything!"
    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    mock_agent_client.ainvoke = AsyncMock(
        return_value=ChatMessage(type="ai", content=RESPONSE, trace_id="trace-ui"),
    )

    messages = list(at.chat_message)
    if messages and messages[0].avatar == "assistant":
        assert messages[0].markdown[0].value.startswith(WELCOME_START)
        messages = messages[1:]
    else:
        assert messages == []

    at.sidebar.toggle[0].set_value(False)  # Use Streaming = False
    at.chat_input[0].set_value(PROMPT).run()
    print(at)
    messages = list(at.chat_message)
    if messages and messages[0].avatar == "assistant":
        messages = messages[1:]
    assert messages[0].avatar == "user"
    assert messages[0].markdown[0].value == PROMPT
    assert messages[1].avatar == "assistant"
    assert messages[1].markdown[0].value == RESPONSE
    assert at.session_state.trace_id == "trace-ui"
    assert mock_agent_client.trace_id == "trace-ui"
    assert not at.exception


def test_app_settings(mock_agent_client):
    """Test the full app - happy path"""
    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["user_id"] = "1234"
    at.run()

    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    responses = [
        ChatMessage(type="ai", content=RESPONSE, trace_id="trace-app"),
        ChatMessage(type="ai", content="Second answer", trace_id="trace-next"),
    ]

    async def invoke_side_effect(*args, **kwargs):
        message = responses.pop(0)
        if message.trace_id == "trace-next":
            assert mock_agent_client.trace_id == "trace-app"
        return message

    mock_agent_client.ainvoke = AsyncMock(side_effect=invoke_side_effect)

    at.sidebar.toggle[0].set_value(False)  # Use Streaming = False
    assert at.sidebar.selectbox[0].value == "gpt-4o"
    assert mock_agent_client.agent == "test-agent"
    at.sidebar.selectbox[0].set_value("gpt-4o-mini")
    at.sidebar.selectbox[1].set_value("chatbot")
    at.chat_input[0].set_value(PROMPT).run()
    assert at.session_state.trace_id == "trace-app"
    assert mock_agent_client.trace_id == "trace-app"

    SECOND_PROMPT = "Do you know any riddles?"
    at.chat_input[0].set_value(SECOND_PROMPT).run()
    print(at)

    # Basic checks
    messages = list(at.chat_message)
    if messages and messages[0].avatar == "assistant":
        messages = messages[1:]
    assert messages[0].avatar == "user"
    assert messages[0].markdown[0].value == PROMPT
    assert messages[1].avatar == "assistant"
    assert messages[1].markdown[0].value == RESPONSE
    assert messages[2].markdown[0].value == SECOND_PROMPT
    assert messages[3].markdown[0].value == "Second answer"
    assert at.session_state.trace_id == "trace-next"
    assert mock_agent_client.trace_id == "trace-next"

    # Check the args match the settings
    assert mock_agent_client.agent == "chatbot"
    first_call = mock_agent_client.ainvoke.await_args_list[0]
    assert first_call.kwargs["message"] == PROMPT
    assert first_call.kwargs["model"] == OpenAIModelName.GPT_4O_MINI
    assert first_call.kwargs["thread_id"] == at.session_state.thread_id
    second_call = mock_agent_client.ainvoke.await_args_list[1]
    assert second_call.kwargs["message"] == SECOND_PROMPT
    assert not at.exception


def test_app_thread_id_history(mock_agent_client):
    """Test the thread_id is generated"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # Reset and set thread_id
    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["thread_id"] = "1234"
    HISTORY = [
        ChatMessage(type="human", content="What is the weather?"),
        ChatMessage(type="ai", content="The weather is sunny."),
    ]
    mock_agent_client.get_history.return_value = ChatHistory(messages=HISTORY)
    at.run()
    print(at)
    assert at.session_state.thread_id == "1234"
    mock_agent_client.get_history.assert_called_with(thread_id="1234")
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == "What is the weather?"
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == "The weather is sunny."
    assert not at.exception


def test_app_feedback(mock_agent_client):
    """TODO: Can't figure out how to interact with st.feedback"""

    pass


@pytest.mark.asyncio
async def test_app_streaming(mock_agent_client):
    """Test the app with streaming enabled - including tool messages"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # Setup mock streaming response
    PROMPT = "What is 6 * 7?"
    ai_with_tool = ChatMessage(
        type="ai",
        content="",
        tool_calls=[{"name": "calculator", "id": "test_call_id", "args": {"expression": "6 * 7"}}],
    )
    tool_message = ChatMessage(type="tool", content="42", tool_call_id="test_call_id")
    final_ai_message = ChatMessage(type="ai", content="The answer is 42", trace_id="trace-stream")

    messages = [ai_with_tool, tool_message, final_ai_message]

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            if isinstance(m, ChatMessage) and m.trace_id:
                mock_agent_client.trace_id = m.trace_id
            yield m

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)  # Use Streaming = True
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    messages = list(at.chat_message)
    if messages and messages[0].avatar == "assistant":
        messages = messages[1:]
    assert messages[0].avatar == "user"
    assert messages[0].markdown[0].value == PROMPT
    response = messages[1]
    tool_status = response.status[0]
    assert response.avatar == "assistant"
    assert tool_status.label == "Tool Call: calculator"
    assert tool_status.icon == ":material/check:"
    assert tool_status.markdown[0].value == "Input:"
    assert tool_status.json[0].value == '{"expression": "6 * 7"}'
    assert tool_status.markdown[1].value == "Output:"
    assert tool_status.markdown[2].value == "42"
    assert response.markdown[-1].value == "The answer is 42"
    assert at.session_state.trace_id == "trace-stream"
    assert mock_agent_client.trace_id == "trace-stream"
    assert not at.exception


@pytest.mark.asyncio
async def test_app_init_error(mock_agent_client):
    """Test the app with an error in the agent initialization"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # Setup mock streaming response
    PROMPT = "What is 6 * 7?"
    mock_agent_client.astream.side_effect = AgentClientError("Error connecting to agent")

    at.toggle[0].set_value(True)  # Use Streaming = True
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    messages = list(at.chat_message)
    assert messages[0].avatar == "assistant"
    assert messages[1].avatar == "user"
    assert messages[1].markdown[0].value == PROMPT
    assert at.error[0].value == "Error generating response: Error connecting to agent"
    assert not at.exception


def test_app_new_chat_btn(mock_agent_client):
    at = AppTest.from_file("../../src/streamlit_app.py").run()
    thread_id_a = at.session_state.thread_id
    at.session_state.trace_id = "trace-old"
    mock_agent_client.trace_id = "trace-old"

    at.sidebar.button[0].click().run()

    assert at.session_state.thread_id != thread_id_a
    assert at.session_state.trace_id is None
    assert mock_agent_client.trace_id is None
    assert not at.exception

