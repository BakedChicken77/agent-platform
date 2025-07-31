
import asyncio
from typing import cast
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

from agents import DEFAULT_AGENT, get_agent  # noqa: E402

# The default agent uses StateGraph.compile() which returns CompiledStateGraph
agent = cast(CompiledStateGraph, get_agent(DEFAULT_AGENT))
agent1 = cast(CompiledStateGraph, get_agent('chatbot'))
agent2 = cast(CompiledStateGraph, get_agent('research-assistant'))
agent3 = cast(CompiledStateGraph, get_agent('rag-assistant'))
agent4 = cast(CompiledStateGraph, get_agent('command-agent'))
agent5 = cast(CompiledStateGraph, get_agent('bg-task-agent'))
agent6 = cast(CompiledStateGraph, get_agent('langgraph-supervisor-agent'))
agent7 = cast(CompiledStateGraph, get_agent('knowledge-base-agent'))
agent8 = cast(CompiledStateGraph, get_agent('agent-team'))
agent9 = cast(CompiledStateGraph, get_agent('agent-team2'))

async def main() -> None:
    

    agent1.get_graph(xray=True).draw_png("chatbot.png")
    agent2.get_graph(xray=True).draw_png("research-assistant.png")
    agent3.get_graph(xray=True).draw_png("rag-assistant.png")
    agent4.get_graph(xray=True).draw_png("command-agent.png")
    agent5.get_graph(xray=True).draw_png("bg-task-agent.png")
    agent6.get_graph(xray=True).draw_png("langgraph-supervisor-agent.png")
    agent7.get_graph(xray=True).draw_png("knowledge-base-agent.png")
    agent8.get_graph(xray=True).draw_png("agent-team.png")
    agent9.get_graph(xray=True).draw_png("agent-team2.png")





    # inputs: MessagesState = {
    #     "messages": [HumanMessage("Find me a recipe for chocolate chip cookies")]
    # }
    # result = await agent.ainvoke(
    #     input=inputs,
    #     config=RunnableConfig(configurable={"thread_id": uuid4()}),
    # )
    # result["messages"][-1].pretty_print()

    # # Draw the agent graph as png
    # # requires:
    # # brew install graphviz
    # # export CFLAGS="-I $(brew --prefix graphviz)/include"
    # # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
    # # pip install pygraphviz
    # #
    # agent.get_graph().draw_png("agent_diagram.png")


if __name__ == "__main__":
    asyncio.run(main())

