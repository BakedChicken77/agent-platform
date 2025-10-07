## src/agents/agent_registry.py

from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel



from agents.chatbot import chatbot
from agents.universal_supervisor_agent import universal_supervisor_agent


from schema import AgentInfo

# Register file tools module so agents can import/bind them where needed
try:
    from agents.tools_files import ListUserFiles, ReadUserFile  # noqa: F401
except Exception:
    # tools remain optional; agents can bind when present
    pass



DEFAULT_AGENT = "General Chatbot"


# Type alias to handle LangGraph's different agent patterns
# - @entrypoint functions return Pregel
# - StateGraph().compile() returns CompiledStateGraph
AgentGraph = CompiledStateGraph | Pregel


@dataclass
class Agent:
    description: str
    graph: AgentGraph


agents: dict[str, Agent] = {
    chatbot.name: Agent(
        description="A simple chatbot.", 
        graph=chatbot
    ),
    universal_supervisor_agent.name: Agent(
        description="A JACSKE Documentation and Python Coding Agent ",
        graph=universal_supervisor_agent,
    ),
}


def get_agent(agent_id: str) -> AgentGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]

