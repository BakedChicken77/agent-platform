
"""
Hierarchical Agent Team
=======================

A three-layer team composed of:
1. **Research subgraph**: search & web-scraping agents.
2. **Writing subgraph**: note-taker, doc-writer, chart-generator.
3. **Top-level supervisor** orchestrating the two subgraphs.

This follows the "Hierarchical Agent Teams" tutorial, using LangGraph and LangChain's create_react_agent.
"""

from typing import Literal, TypedDict, List, Optional, Dict

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.document_loaders import WebBaseLoader
# from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from tempfile import TemporaryDirectory
from pathlib import Path

from core import get_model, settings

from agents.tools_writing import create_outline, read_document, write_document, edit_document, python_repl_tool
from agents.tools import database_search, get_full_doc_text




# ---------------------------------------------------------------------------
# ••• Tools for research subgraph
# ---------------------------------------------------------------------------

# tavily_tool = TavilySearch(max_results=5)

# @tool
# def scrape_webpages(urls: List[str]) -> str:
#     """Scrape provided web pages for detailed information."""
#     loader = WebBaseLoader(urls)
#     docs = loader.load()
#     return "\n\n".join(
#         [
#             f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
#             for doc in docs
#         ]
#     )

# ---------------------------------------------------------------------------
# ••• Tools for document-writing subgraph
# ---------------------------------------------------------------------------

# _TEMP_DIRECTORY = PythonREPL().run  # dummy to silence flake; real TemporaryDirectory import omitted
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)
# Tools create_outline, read_document, write_document, edit_document, python_repl_tool
# would be imported from the existing codebase as needed

# For brevity assume these tools are imported:
# from agents.tools import create_outline, read_document, write_document, edit_document
# and python_repl_tool from utilities

# ---------------------------------------------------------------------------
# ••• State definitions
# ---------------------------------------------------------------------------

class TeamState(MessagesState, total=False):
    """Shared conversation state."""
    next: str

# ---------------------------------------------------------------------------
# ••• Supervisor factory
# ---------------------------------------------------------------------------

def make_supervisor_node(llm: BaseChatModel, members: List[str]):
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor managing the workers: "
        f"{', '.join(members)}. After reading the conversation decide which "
        "worker should act next. Respond ONLY with the worker name, or `FINISH`. "
        "Use python_repl to write and run python code to complete tasks."
    )
    class Router(TypedDict):
        next: Literal[*options]

    def supervisor_node(state: TeamState) -> Command[Literal[*members, "__end__"]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        return Command(goto=goto, update={"next": goto})

    return supervisor_node

# ---------------------------------------------------------------------------
# ••• Research subgraph
# ---------------------------------------------------------------------------

llm = get_model(settings.DEFAULT_MODEL)
search_agent = create_react_agent(llm, tools=[database_search])
get_full_doc_text_agent = create_react_agent(llm, tools=[get_full_doc_text])

async def semantic_search_node(state: TeamState) -> Command[Literal["research_supervisor"]]:
    result = await search_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="research_supervisor",
    )

async def get_full_doc_text_node(state: TeamState) -> Command[Literal["research_supervisor"]]:
    result = await get_full_doc_text_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="research_supervisor",
    )

research_supervisor = make_supervisor_node(llm, ["semantic_search", "get_full_doc_text"])

research_builder = StateGraph(TeamState)
research_builder.add_node("research_supervisor", research_supervisor)
research_builder.add_node("semantic_search", semantic_search_node)
research_builder.add_node("get_full_doc_text", get_full_doc_text_node)
research_builder.add_edge(START, "research_supervisor")
research_graph = research_builder.compile()

# ---------------------------------------------------------------------------
# ••• Document-writing subgraph
# ---------------------------------------------------------------------------

# Assume create_outline, read_document, write_document, edit_document, python_repl_tool imported

doc_writer_agent = create_react_agent(
    llm,
    tools=[write_document, edit_document, read_document],
    prompt=(
        "You can read, write and edit documents based on note-taker's outlines. "
        "Don't ask follow-up questions."
    ),
    name="doc_writer_agent"
)
note_taker_agent = create_react_agent(
    llm,
    tools=[create_outline, read_document],
    prompt=(
        "You can read documents and create outlines for the doc writer. "
        "Don't ask follow-up questions."
    ),
    name="note_taker_agent"
)
chart_agent = create_react_agent(
    llm, 
    tools=[read_document, python_repl_tool],
    name="chart_agent"
)

repl_agent = create_react_agent(
    llm,
    tools=[python_repl_tool],
    name="python_repl_agent",
    prompt="You are a Python REPL. Execute the user’s code and return only the output."
)

async def note_node(state: TeamState) -> Command[Literal["doc_supervisor"]]:
    result = await note_taker_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="doc_supervisor",
    )

async def doc_node(state: TeamState) -> Command[Literal["doc_supervisor"]]:
    result = await doc_writer_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="doc_supervisor",
    )

async def chart_node(state: TeamState) -> Command[Literal["doc_supervisor"]]:
    result = await chart_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content,name="chart_generator")]},
        goto="doc_supervisor",
    )

async def repl_node(state: TeamState) -> Command[Literal["top_supervisor"]]:
    result = await repl_agent.ainvoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content)]},
        goto="top_supervisor",
    )

doc_supervisor = make_supervisor_node(llm, ["note_taker", "doc_writer", "chart_generator"])

doc_builder = StateGraph(TeamState)
doc_builder.add_node("doc_supervisor", doc_supervisor)
doc_builder.add_node("note_taker", note_node)
doc_builder.add_node("doc_writer", doc_node)
doc_builder.add_node("chart_generator", chart_node)
doc_builder.add_edge(START, "doc_supervisor")
doc_graph = doc_builder.compile()

# ---------------------------------------------------------------------------
# ••• Top-level composition
# ---------------------------------------------------------------------------

top_supervisor = make_supervisor_node(llm, ["research_team", "writing_team", "python_repl"])

async def call_research_team(state: TeamState) -> Command[Literal["top_supervisor"]]:
    # feed last research output into research_graph
    response = await research_graph.ainvoke({"messages": state["messages"][-1]})
    return Command(
        update={"messages": [HumanMessage(content=response["messages"][-1].content)]},
        goto="top_supervisor",
    )

async def call_writing_team(state: TeamState) -> Command[Literal["top_supervisor"]]:
    response = await doc_graph.ainvoke({"messages": state["messages"][-1]})
    return Command(
        update={"messages": [HumanMessage(content=response["messages"][-1].content)]},
        goto="top_supervisor",
    )

builder = StateGraph(TeamState)
builder.add_node("top_supervisor", top_supervisor)
builder.add_node("research_team", call_research_team)
builder.add_node("writing_team", call_writing_team)
builder.add_node("python_repl", repl_node)
builder.add_edge(START, "top_supervisor")
agent_team2 = builder.compile()
agent_team2.name = "hierarchical-agent-team"


def visualize_graph(graph, filename='workflow.dot'):
    dot = graphviz.Digraph()

    # Add nodes
    for node in graph.nodes:
        dot.node(node, node)

    # Add edges
    for edge in graph.edges:
        dot.edge(edge[0], edge[1])

    # Save to file and render as PNG
    dot.save(filename)
    dot.render(filename, format='png', cleanup=True)

    dot = graphviz.Digraph()

    # # Custom labels for nodes
    # node_labels = {
    #     "document_search": "Document Search",
    #     "generate": "Generate",
    #     "transform_query": "Transform Query",
    #     "web_search": "Web Search",
    #     "finalize_response": "Finalize Response"
    # }

    node_labels = False
    if node_labels:
        # Add nodes with custom labels
        for node in graph.nodes:
            label = node_labels.get(node, node)  # Default to node ID if no custom label
            dot.node(node, label)
    else:
        # Add nodes
        for node in graph.nodes:
            dot.node(node, node)

    # Add edges
    for edge in graph.edges:
        dot.edge(edge[0], edge[1])

    # Handle conditional edges
    for start_node, branches in graph.branches.items():
        for condition_func, branch in branches.items():
            for condition, end_node in branch.ends.items():
                edge_label = f"{condition_func} -> {condition}"
                # Modify the edge label to remove the redundant part
                dot.edge(start_node, end_node, label=condition_func, style="dashed")

    # Save to file and render as PNG
    dot.save(filename)
    dot.render(filename, format='png', cleanup=True)

# Function to visualize without executing the workflow
# async def visualize_only():
#     import graphviz
#     visualize_graph(workflow)

if __name__ == "__main__":
    import graphviz
    visualize_graph(builder)
