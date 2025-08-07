import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from core import settings
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.checkpoint.memory import InMemorySaver  # Added for state persistence

logger = logging.getLogger(__name__)


class IdeationState(MessagesState, total=False):
    """State object for the ideation workflow."""

    phase: int
    approval_state: Literal["pending", "approved", "changes_requested"]
    last_output: str | None


PHASE_QUESTIONS = {
    1: "Phase 1 - What idea would you like to propose?",
    2: "Phase 2 - Brainstorm potential use cases and benefits.",
    3: "Phase 3 - Outline a high-level execution roadmap.",
    4: "Phase 4 - List risks, blockers or open questions.",
}

FORMAT_TEMPLATE = (
    "Here is what you said:\n{user_input}\n\nHow does this sound?"
)

APPROVAL_TEMPLATE = (
    "The user responded: '{response}'.\n"
    "If they approve, reply only with 'approved'. "
    "Otherwise reply with 'changes_requested'."
)


async def ask_phase(state: IdeationState, config: RunnableConfig) -> IdeationState:
    """Ask the phase-specific question and capture user input."""
    phase = state.get("phase", 1)
    question = PHASE_QUESTIONS.get(phase, "")
    logger.debug("Asking question for phase %s", phase)
    # Ask the question and pause for user input
    user_reply = interrupt(question)
    return {
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=user_reply)],
        "approval_state": "pending",
    }


async def format_phase(state: IdeationState, config: RunnableConfig) -> IdeationState:
    """Summarize user input, ask for confirmation, and capture response."""
    human_messages = [m.content for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    user_input = human_messages[-1] if human_messages else ""
    logger.debug("Formatting user input: %s", user_input)
    formatted = FORMAT_TEMPLATE.format(user_input=user_input)
    # Show formatted summary and wait for user feedback
    interrupt(formatted)
    response_messages = state.get("messages", []) + [AIMessage(content=formatted)]
    return {
        **state,
        "messages": response_messages,
        "last_output": formatted,
    }


async def check_approval(state: IdeationState, config: RunnableConfig) -> IdeationState:
    """Prompt user for approval of last output."""
    prompt = f"Do you approve the following summary?\n\n{state.get('last_output', '')}"
    logger.debug("Checking approval prompt: %s", prompt)
    # Pause for approval response
    response = interrupt(prompt)
    approval = "approved" if response.strip().lower().startswith("y") else "changes_requested"
    return {
        **state,
        "approval_state": approval,
    }


async def increment_phase(state: IdeationState, config: RunnableConfig) -> IdeationState:
    """Advance to the next phase if approved."""
    phase = state.get("phase", 1)
    logger.debug("Current phase %s with state %s", phase, state.get("approval_state"))
    if state.get("approval_state") == "approved":
        phase += 1
    return {
        **state,
        "phase": phase,
        "approval_state": "pending",
    }


# Build graph
agent = StateGraph(IdeationState)
agent.add_node("ask_phase", ask_phase)
agent.add_node("format_phase", format_phase)
agent.add_node("check_approval", check_approval)
agent.add_node("increment_phase", increment_phase)

agent.set_entry_point("ask_phase")
agent.add_edge("ask_phase", "format_phase")
agent.add_edge("format_phase", "check_approval")

# Route based on approval state

def route_approval(state: IdeationState) -> Literal["format_phase", "increment_phase"]:
    return (
        "increment_phase"
        if state.get("approval_state") == "approved"
        else "format_phase"
    )

agent.add_conditional_edges("check_approval", route_approval, {
    "increment_phase": "increment_phase",
    "format_phase": "format_phase",
})


def route_continue(state: IdeationState) -> Literal["ask_phase", "end"]:
    return "ask_phase" if state.get("phase", 1) <= 4 else "end"

agent.add_conditional_edges("increment_phase", route_continue, {
    "ask_phase": "ask_phase",
    "end": END,
})

# Compile graph for human-in-the-loop
interactive_ideation_agent = agent.compile(
    checkpointer=InMemorySaver(),
)
interactive_ideation_agent.name = "BEX-Idea-Agent"


def get_ideation_agent_graph() -> StateGraph:
    """Return the compiled ideation agent graph."""
    return interactive_ideation_agent
