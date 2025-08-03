# Adding a Custom Agent and Configuring Status Messages

This guide summarizes how to extend the toolkit with your own agent and display status updates in the Streamlit UI.

## Adding a New Agent

1. Create a new file in `src/agents/` containing your LangGraph graph. You can start from `research_assistant.py` or `chatbot.py`.
2. Register the agent in `src/agents/agent_registry.py` by inserting it into the `agents` dictionary with a description. Optionally update `DEFAULT_AGENT`.
3. Update `src/streamlit_app.py` if your agent needs a custom welcome message or other UI tweaks.

The README describes these steps under *Building or customizing your own agent*:

```text
 1. Add your new agent to the `src/agents` directory. You can copy `research_assistant.py` or `chatbot.py` and modify it to change the agent's behavior and tools.
 1. Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`. Your agent can be called by `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
 1. Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.
```

## Configuring Status Messages

Agents can send progress or task updates using `CustomData` messages. The `bg_task_agent` provides an example via the `Task` helper:

```python
class Task:
    def __init__(self, task_name: str, writer: StreamWriter | None = None) -> None:
        self.name = task_name
        self.id = str(uuid4())
        self.state: Literal["new", "running", "complete"] = "new"
        self.result: Literal["success", "error"] | None = None
```

`Task.start()`, `write_data()` and `finish()` dispatch `CustomData` objects that encapsulate `TaskData` information. On the Streamlit side, `TaskDataStatus` formats these messages:

```python
class TaskDataStatus:
    def __init__(self) -> None:
        import streamlit as st
        self.status = st.status("")
        self.current_task_data: dict[str, TaskData] = {}

    def add_and_draw_task_data(self, task_data: TaskData) -> None:
        status = self.status
        status_str = f"Task **{task_data.name}** "
        match task_data.state:
            case "new":
                status_str += "has :blue[started]. Input:"
            case "running":
                status_str += "wrote:"
            case "complete":
                if task_data.result == "success":
                    status_str += ":green[completed successfully]. Output:"
                else:
                    status_str += ":red[ended with error]. Output:"
        status.write(status_str)
        status.write(task_data.data)
        status.write("---")
```

You can customize this logic to change how progress is shown or create your own handler. Include `TaskDataStatus` in `streamlit_app.py` to display updates when your agent emits `CustomData` messages.

## Running the App

Follow the Quickstart in the README to launch the service and UI:

```bash
# At least one LLM API key is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
uv sync --frozen
source .venv/bin/activate
python src/run_service.py

# In another shell
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

You can also use Docker:

```bash
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```
