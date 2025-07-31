
"""
Hierarchical Agent Team
=======================

A two‑layer team that first gathers facts with the existing *research_assistant*
sub‑graph and then hands the material to a dedicated writing agent that
crafts the final answer.

The top‑level “supervisor” decides which worker to run next and when the job
is finished, following the pattern shown in the “Hierarchical Agent Teams”
tutorial notebook.
"""

from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from core import get_model, settings
from agents.research_assistant import research_assistant  # existing graph
from langchain_core.messages import AIMessage


# --------------------------------------------------------------------------- #
# ••• State definition                                                        #
# --------------------------------------------------------------------------- #

class TeamState(MessagesState, total=False):
    """Conversation state shared by the team."""
    next: str  # name of the next node selected by the supervisor


# --------------------------------------------------------------------------- #
# ••• Utility: build a supervisor (router) node                               #
# --------------------------------------------------------------------------- #

def _make_supervisor(llm: ChatOpenAI, members: list[str]):
    """
    Build a router that picks the next worker.

    The LLM is asked to answer with one of the *members* names or the token
    `FINISH`.  When it answers `FINISH` the graph transitions to `__end__`.
    """
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor managing the workers: "
        f"{', '.join(members)}.\n"
        "After reading the conversation decide which worker should act next. "
        "Respond **only** with the worker name, or `FINISH` if the task is done."
    )

    class Router(TypedDict):
        next: Literal[*options]

    def _node(state: TeamState) -> Command[Literal[*members, "__end__"]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        choice = llm.with_structured_output(Router).invoke(messages)
        goto = choice["next"]
        if goto == "FINISH":
            goto = END
        return Command(
            goto=goto,
            update={"next": goto, "messages": [AIMessage(content="")]}
        )

    return _node


# --------------------------------------------------------------------------- #
# ••• Worker nodes                                                            #
# --------------------------------------------------------------------------- #

async def _research_worker(state: TeamState) -> Command[Literal["team_supervisor"]]:
    """
    Delegate to the existing `research_assistant` sub‑graph and return its
    final message back to the supervisor.
    """
    response = await research_assistant.ainvoke({"messages": state["messages"]})
    return Command(
        update={"messages": [response["messages"][-1]]},
        goto="team_supervisor",
    )


# A very small “writing” agent: just call the LLM with the conversation so far
_writer_llm = get_model(settings.DEFAULT_MODEL)


def _writer_worker(state: TeamState) -> Command[Literal["team_supervisor"]]:
    """Turn the accumulated research into a clear, concise answer."""
    reply: BaseMessage = _writer_llm.invoke(state["messages"])
    return Command(update={"messages": [reply]}, goto="team_supervisor")


# --------------------------------------------------------------------------- #
# ••• Build and export the compiled graph                                     #
# --------------------------------------------------------------------------- #

def _build_team():
    router_llm = get_model(settings.DEFAULT_MODEL)
    supervisor = _make_supervisor(router_llm, ["research_expert", "writer_worker"])

    builder = StateGraph(TeamState)
    builder.add_node("team_supervisor", supervisor)
    builder.add_node("research_expert", _research_worker)
    builder.add_node("writer_worker", _writer_worker)

    builder.add_edge(START, "team_supervisor")          # entry‑point

    graph = builder.compile()
    graph.name = "hierarchical-agent-team"
    return graph


agent_team = _build_team()

