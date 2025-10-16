import truststore
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from agents.instrumentation import configure_agent, with_langfuse_span
from core import get_model, settings
from langgraph.func import entrypoint

truststore.inject_into_ssl()

@entrypoint()
@with_langfuse_span("chatbot")
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
    return entrypoint.final(
        value={"messages": [response]}, save={"messages": messages + [response]}
    )

chatbot.name = "General Chatbot"
chatbot = configure_agent(
    chatbot,
    agent_id=chatbot.name,
    agent_kind="entrypoint",
    description="A simple chatbot.",
)
