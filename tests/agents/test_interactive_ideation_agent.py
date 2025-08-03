"""Unit tests for interactive ideation agent."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture
def ideation_module(monkeypatch: pytest.MonkeyPatch):
    """Load the interactive_ideation_agent module with stubbed dependencies."""
    root = pathlib.Path(__file__).resolve().parents[2]

    # Ensure local 'langgraph' directory does not shadow the package installation.
    for p in ("", str(root)):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.append(str(root / "src"))

    # Stub minimal settings object expected by the module.
    settings = types.SimpleNamespace(
        AZURE_OPENAI_ENDPOINT="https://example.com",
        AZURE_OPENAI_DEPLOYMENT_NAME="dep",
        AZURE_OPENAI_API_VERSION="2024-06-01",
    )
    core = types.ModuleType("core")
    core.settings = settings
    monkeypatch.setitem(sys.modules, "core", core)

    # Stub out langgraph cache dependency used during import.
    cache_base = types.ModuleType("langgraph.cache.base")
    class BaseCache:  # pragma: no cover - placeholder for import
        pass
    cache_base.BaseCache = BaseCache
    monkeypatch.setitem(sys.modules, "langgraph.cache.base", cache_base)

    # Provide a fallback interrupt_after method if using older langgraph versions.
    from langgraph.graph import StateGraph

    if not hasattr(StateGraph, "interrupt_after"):
        def interrupt_after(self, node: str) -> None:  # pragma: no cover - simple stub
            self._interrupt_after = getattr(self, "_interrupt_after", []) + [node]
        StateGraph.interrupt_after = interrupt_after  # type: ignore[attr-defined]

    path = root / "src" / "agents" / "interactive_ideation_agent.py"
    spec = importlib.util.spec_from_file_location("interactive_ideation_agent", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_ask_phase_default_question(ideation_module):
    result = await ideation_module.ask_phase({"phase": 1}, {})
    assert result["messages"][0].content == ideation_module.PHASE_QUESTIONS[1]
    assert result["approval_state"] == "pending"


@pytest.mark.asyncio
async def test_format_phase_summarises_user_input(ideation_module):
    state = {"messages": [HumanMessage(content="first"), HumanMessage(content="second")]}  # noqa: E501
    result = await ideation_module.format_phase(state, {})
    expected = ideation_module.FORMAT_TEMPLATE.format(user_input="second")
    assert result["messages"][0].content == expected
    assert result["last_output"] == expected


@pytest.mark.asyncio
async def test_check_approval_approved(monkeypatch: pytest.MonkeyPatch, ideation_module):
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, messages: Any) -> AIMessage:
            return AIMessage(content="approved")

    monkeypatch.setattr(ideation_module, "AzureChatOpenAI", MockModel)
    state = {"messages": [HumanMessage(content="Yes")]}  # user response
    result = await ideation_module.check_approval(state, {})
    assert result["approval_state"] == "approved"


@pytest.mark.asyncio
async def test_check_approval_changes_requested(monkeypatch: pytest.MonkeyPatch, ideation_module):
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, messages: Any) -> AIMessage:
            return AIMessage(content="maybe")

    monkeypatch.setattr(ideation_module, "AzureChatOpenAI", MockModel)
    state = {"messages": [HumanMessage(content="No")]}  # human reply irrelevant
    result = await ideation_module.check_approval(state, {})
    assert result["approval_state"] == "changes_requested"


@pytest.mark.asyncio
async def test_increment_phase_handles_approval(ideation_module):
    state = {"phase": 2, "approval_state": "approved"}
    result = await ideation_module.increment_phase(state, {})
    assert result["phase"] == 3
    assert result["approval_state"] == "pending"


@pytest.mark.asyncio
async def test_increment_phase_no_approval(ideation_module):
    state = {"phase": 2, "approval_state": "changes_requested"}
    result = await ideation_module.increment_phase(state, {})
    assert result["phase"] == 2
    assert result["approval_state"] == "pending"


def test_routing_helpers(ideation_module):
    assert ideation_module.route_approval({"approval_state": "approved"}) == "increment_phase"
    assert ideation_module.route_approval({"approval_state": "changes_requested"}) == "format_phase"
    assert ideation_module.route_continue({"phase": 4}) == "ask_phase"
    assert ideation_module.route_continue({"phase": 5}) == "end"


@pytest.mark.asyncio
async def test_full_graph_run(monkeypatch: pytest.MonkeyPatch, ideation_module):
    """Simulate a full run where the user approves each phase."""
    class ApproveModel:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, messages: Any) -> AIMessage:
            return AIMessage(content="approved")

    monkeypatch.setattr(ideation_module, "AzureChatOpenAI", ApproveModel)
    graph = ideation_module.get_ideation_agent_graph()
    final_state = await graph.ainvoke({"messages": [], "phase": 1})
    assert final_state["phase"] == 5
    assert final_state["approval_state"] == "pending"
    assert len(final_state["messages"]) >= 8  # ensures all phases ran


@pytest.mark.asyncio
async def test_format_phase_handles_missing_input(ideation_module):
    result = await ideation_module.format_phase({"messages": []}, {})
    assert "How does this sound" in result["messages"][0].content


def test_interrupt_registration(ideation_module):
    # interrupt_after is patched to collect nodes in _interrupt_after
    assert "ask_phase" in ideation_module.agent._interrupt_after
    assert "format_phase" in ideation_module.agent._interrupt_after
