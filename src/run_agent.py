import asyncio
from typing import cast
from uuid import uuid4

from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage
# from langchain_core.runnables import RunnableConfig
# from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from pathlib import Path

load_dotenv()

from agents import get_agent, get_all_agent_info  # noqa: E402

folder_path = Path(r".\workflow_diagrams")

agents_info = get_all_agent_info()
agents = []
for agent in agents_info:
    agents.append(cast(CompiledStateGraph, get_agent(agent.key)))


async def main() -> None:
    for agent in agents:
        file_path = folder_path / Path(f"{agent.name}.png")
        agent.get_graph(xray=True).draw_png(file_path)



        # if agent.name == "mtp_mapper_workflow":
            
            # inputs: MessagesState = {
            #     "messages": [HumanMessage("AS-18 i) Pod software will be based on the P5 Spectrum Relocation software, which includes the 114 player adaptive mode.")]
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
