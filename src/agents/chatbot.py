from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint

from core import get_model, settings
from core.langgraph import instrument_langgraph_node, persist_langfuse_state
import truststore
truststore.inject_into_ssl()

@entrypoint()
@instrument_langgraph_node("chatbot")
async def chatbot(
    inputs: dict[str, list[BaseMessage]],
    *,
    previous: dict[str, list[BaseMessage]],
    config: RunnableConfig,
):
    messages = inputs["messages"]
    if previous:
        messages = previous["messages"] + messages

    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = await model.ainvoke(messages)
    save_state = persist_langfuse_state(
        {"messages": messages + [response]}, config=config, previous=previous
    )
    return entrypoint.final(value={"messages": [response]}, save=save_state)

chatbot.name = "General Chatbot"
