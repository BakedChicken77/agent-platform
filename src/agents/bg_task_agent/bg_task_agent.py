
import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import StreamWriter

from agents.bg_task_agent.task import Task
from core import get_model, settings
from core.langgraph import (
    build_langfuse_runtime,
    emit_runtime_event,
    ensure_langfuse_state,
    instrument_langgraph_node,
    runtime_metadata,
)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: state["messages"],
        name="StateModifier",
    )
    return preprocessor | model  # type: ignore[return-value]


@instrument_langgraph_node("bg_task.model")
async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    ensure_langfuse_state(state, config=config)
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    result: AgentState = {"messages": [response]}
    ensure_langfuse_state(result, config=config, previous=state)
    return result

def _emit_bg_heartbeat(runtime: dict[str, Any], status: str, **extra: Any) -> None:
    if not runtime:
        return
    metadata = runtime_metadata(runtime, status=status, **extra)
    emit_runtime_event("agent.bg_task.heartbeat", runtime, metadata=metadata)


@instrument_langgraph_node("bg_task.runner")
async def bg_task(
    state: AgentState,
    writer: StreamWriter,
    config: RunnableConfig | None = None,
) -> AgentState:
    ensure_langfuse_state(state, config=config)
    runtime = build_langfuse_runtime(config=config, state=state)
    _emit_bg_heartbeat(runtime, "queued")

    task1 = Task("Simple task 1...", writer, runtime=runtime)
    task2 = Task("Simple task 2...", writer, runtime=runtime)

    task1.start()
    _emit_bg_heartbeat(runtime, "running", step="task1_started")
    await asyncio.sleep(2)
    task2.start()
    _emit_bg_heartbeat(runtime, "running", step="task2_started")
    await asyncio.sleep(2)
    task1.write_data(data={"status": "Still running..."})
    _emit_bg_heartbeat(runtime, "running", step="task1_update")
    await asyncio.sleep(2)
    task2.finish(result="error", data={"output": 42})
    _emit_bg_heartbeat(runtime, "running", step="task2_finished", result="error")
    await asyncio.sleep(2)
    task1.finish(result="success", data={"output": 42})
    _emit_bg_heartbeat(runtime, "completed")
    result: AgentState = {"messages": []}
    ensure_langfuse_state(result, config=config, previous=state)
    return result


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("bg_task", bg_task)
agent.set_entry_point("bg_task")

agent.add_edge("bg_task", "model")
agent.add_edge("model", END)

bg_task_agent = agent.compile()

