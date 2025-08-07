import pytest
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
import agents.interactive_ideation_agent as agent_module

@pytest.fixture
def config():
    return RunnableConfig(configurable={}, run_id=uuid4(), callbacks=[])

@pytest.mark.asyncio
async def test_ask_phase(monkeypatch, config):
    monkeypatch.setattr(agent_module, 'interrupt', lambda prompt: "TestIdea")
    state = {}
    result = await agent_module.ask_phase(state, config)
    assert result["approval_state"] == "pending"
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == 1
    hm = result["messages"][0]
    assert isinstance(hm, HumanMessage)
    assert hm.content == "TestIdea"

@pytest.mark.asyncio
async def test_format_phase(monkeypatch, config):
    captured = {}
    def fake_interrupt(prompt):
        captured['prompt'] = prompt
    monkeypatch.setattr(agent_module, 'interrupt', fake_interrupt)
    human_input = "UserInput"
    state = {"messages": [HumanMessage(content=human_input)]}
    result = await agent_module.format_phase(state, config)
    expected = agent_module.FORMAT_TEMPLATE.format(user_input=human_input)
    assert captured['prompt'] == expected
    assert result["last_output"] == expected
    am = result["messages"][-1]
    assert isinstance(am, AIMessage)
    assert am.content == expected

@pytest.mark.asyncio
async def test_check_approval(monkeypatch, config):
    monkeypatch.setattr(agent_module, 'interrupt', lambda prompt: "yes, approved")
    state = {"last_output": "summary"}
    result = await agent_module.check_approval(state, config)
    assert result["approval_state"] == "approved"
    monkeypatch.setattr(agent_module, 'interrupt', lambda prompt: "nope")
    result = await agent_module.check_approval(state, config)
    assert result["approval_state"] == "changes_requested"

@pytest.mark.asyncio
async def test_increment_phase_and_preserve_fields(config):
    state = {"phase": 1, "approval_state": "approved", "extra": "value"}
    result = await agent_module.increment_phase(state, config)
    assert result["phase"] == 2
    assert result["approval_state"] == "pending"
    assert result["extra"] == "value"
    state = {"phase": 2, "approval_state": "changes_requested", "extra": "value2"}
    result = await agent_module.increment_phase(state, config)
    assert result["phase"] == 2
    assert result["approval_state"] == "pending"
    assert result["extra"] == "value2"

def test_routing_logic():
    assert agent_module.route_approval({"approval_state": "approved"}) == "increment_phase"
    assert agent_module.route_approval({"approval_state": "changes_requested"}) == "format_phase"
    assert agent_module.route_continue({"phase": 4}) == "ask_phase"
    assert agent_module.route_continue({"phase": 5}) == "end"

@pytest.mark.asyncio
async def test_full_end_to_end_flow(monkeypatch, config):
    responses = iter([
        # Phase 1
        "Idea1", None, "yes",
        # Phase 2
        "Idea2", None, "no", None, "yes",
        # Phase 3
        "Idea3", None, "yes",
        # Phase 4
        "Idea4", None, "yes",
    ])
    monkeypatch.setattr(agent_module, 'interrupt', lambda prompt: next(responses))
    state = {}
    # Phase 1
    state = await agent_module.ask_phase(state, config)
    assert state["messages"][-1].content == "Idea1"
    state = await agent_module.format_phase(state, config)
    assert state["last_output"].startswith("Here is what you said")
    state = await agent_module.check_approval(state, config)
    assert state["approval_state"] == "approved"
    state = await agent_module.increment_phase(state, config)
    assert state["phase"] == 2
    # Phase 2 initial cycle
    state = await agent_module.ask_phase(state, config)
    assert state["messages"][-1].content == "Idea2"
    state = await agent_module.format_phase(state, config)
    assert state["last_output"].startswith("Here is what you said")
    state = await agent_module.check_approval(state, config)
    assert state["approval_state"] == "changes_requested"
    # Phase 2 re-format cycle
    state = await agent_module.format_phase(state, config)
    assert isinstance(state["last_output"], str)
    state = await agent_module.check_approval(state, config)
    assert state["approval_state"] == "approved"
    state = await agent_module.increment_phase(state, config)
    assert state["phase"] == 3
    # Phase 3
    state = await agent_module.ask_phase(state, config)
    state = await agent_module.format_phase(state, config)
    state = await agent_module.check_approval(state, config)
    assert state["approval_state"] == "approved"
    state = await agent_module.increment_phase(state, config)
    assert state["phase"] == 4
    # Phase 4
    state = await agent_module.ask_phase(state, config)
    state = await agent_module.format_phase(state, config)
    state = await agent_module.check_approval(state, config)
    assert state["approval_state"] == "approved"
    state = await agent_module.increment_phase(state, config)
    assert state["phase"] == 5
    # Final assertions
    assert state["phase"] == 5
    assert state["approval_state"] == "pending"
    assert state["messages"][-2].content == "Idea4"
    assert state["messages"][-1].content == agent_module.FORMAT_TEMPLATE.format(user_input="Idea4")
