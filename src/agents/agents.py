"""Compatibility shim for legacy imports expecting `agents.agents`.

This module re-exports the agent registry dictionary so older code and tests
can patch `agents.agents.agents` without importing the full service module.
"""

from agents.agent_registry import (
    Agent,
    AgentGraph,
    DEFAULT_AGENT,
    agents,
    get_agent,
    get_all_agent_info,
)

__all__ = [
    "Agent",
    "AgentGraph",
    "DEFAULT_AGENT",
    "agents",
    "get_agent",
    "get_all_agent_info",
]
