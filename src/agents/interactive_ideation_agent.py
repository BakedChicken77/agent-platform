import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI

from core import settings
from langgraph.graph import END, MessagesState, StateGraph

logger = logging.getLogger(__name__)


class IdeationState(MessagesState, total=False):
    """State object for the ideation workflow."""

    phase: int
    approval_state: Literal["pending", "approved", "changes_requested"]
    last_output: str | None


PHASE_QUESTIONS = {
    1: "Phase 1 – What idea would you like to propose?",
    2: "Phase 2 – Brainstorm potential use cases and benefits.",
    3: "Phase 3 – Outline a high-level execution roadmap.",
    4: "Phase 4 – List risks, blockers or open questions.",
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
    """Ask the phase-specific question."""
    phase = state.get("phase", 1)
    question = PHASE_QUESTIONS.get(phase, "")
    logger.debug("Asking question for phase %s", phase)
    return {"messages": [AIMessage(content=question)], "approval_state": "pending"}


async def format_phase(state: IdeationState, config: RunnableConfig) -> IdeationState:
    """Summarize user input and ask for confirmation."""
    human_messages = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    user_input = human_messages[-1] if human_messages else ""
    logger.debug("Formatting user input: %s", user_input)
    formatted = FORMAT_TEMPLATE.format(user_input=user_input)
    return {"messages": [AIMessage(content=formatted)], "last_output": formatted}


async def check_approval(state: IdeationState, config: RunnableConfig) -> IdeationState:
    """Determine if the user approved the previous output."""
    human_messages = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    response = human_messages[-1] if human_messages else ""
    logger.debug("Checking approval for response: %s", response)
    prompt = SystemMessage(content=APPROVAL_TEMPLATE.format(response=response))
    model = AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        temperature=0.0,
    )
    result = await model.ainvoke([prompt])
    output = result.content.strip().lower()
    approval = "approved" if output.startswith("approved") else "changes_requested"
    logger.debug("Approval result: %s", approval)
    return {"approval_state": approval}


async def increment_phase(state: IdeationState, config: RunnableConfig) -> IdeationState:
    """Advance to the next phase if approved."""
    phase = state.get("phase", 1)
    logger.debug("Current phase %s with state %s", phase, state.get("approval_state"))
    if state.get("approval_state") == "approved":
        phase += 1
    return {"phase": phase, "approval_state": "pending"}


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

agent.interrupt_after("ask_phase")
agent.interrupt_after("format_phase")

interactive_ideation_agent = agent.compile()
interactive_ideation_agent.name = "interactive-ideation-agent"


def get_ideation_agent_graph() -> StateGraph:
    """Return the compiled ideation agent graph."""
    return interactive_ideation_agent
