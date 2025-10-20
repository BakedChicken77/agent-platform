# Agents Module (src/agents)

This directory contains the definitions of all AI agents provided by the platform. An **agent** in this context is a self-contained conversational or task-oriented AI logic built with the LangGraph framework. Each agent is implemented as a Python module (or package) that defines how the agent processes inputs, maintains state, and uses tools to produce outputs.

## Purpose

The agents module is the core of the application’s intelligence. It defines multiple agents with different capabilities, all conforming to a common interface so they can be served by the API. The agents are realized as LangGraph state graphs or chains, combining LLM calls and tool executions. By organizing them here, we can easily add or modify agent behaviors independently of the service and UI.

## Agents Overview

Currently defined agents include:

- **General Chatbot (`chatbot.py`):** A simple open-domain conversational agent. This is the **default agent** (named "General Chatbot") for basic Q&A and chit-chat. It takes user messages and responds using an LLM, maintaining a conversational history.
- **Coding Expert (`coding_agent.py`):** An agent specialized in **Python coding and data science assistance**. It can execute Python code via an embedded REPL tool and return results or plots. For example, it can run code using libraries like pandas, NumPy, or Plotly and return either data outputs or visualizations. It is registered under the name "coding_expert". This agent uses a tool `python_repl` to execute code within a sandboxed environment (with certain libraries pre-imported), and includes guidelines in its prompt for how to format outputs (like special handling of Plotly JSON or matplotlib images).
- **Documentation & Coding Supervisor (`universal_supervisor_agent.py`):** A **multi-agent supervisor** that orchestrates specialized expert agents. This agent (named "JACSKE_Agent_Coding" internally) manages three sub-agents:
  - a *research expert* for documentation lookup,
  - a *math expert* for calculations,
  - the *coding expert* for programming tasks.
  The supervisor decides which expert should handle a given user query and delegates the task, then composes the final answer. It uses the **LangGraph Supervisor** pattern to enable agent collaboration. This agent is useful for complex queries that may involve reading technical documents, doing math, and writing code. (JACSKE is a project-specific name related to the documentation domain it was built for.)
- **Excel Assistant (`excel_agent.py`):** An advanced agent that demonstrates **tool usage and safety interrupts**. It helps parse text and write results to an Excel file via a `write_to_excel` tool. It incorporates **LlamaGuard** content moderation – if the AI’s answer might be unsafe, it returns a warning instead. It also handles multi-step interactions: if needed information is missing, it can interrupt itself to ask the user a follow-up question (using the LangGraph `RemainingSteps` logic). This agent is named "excel_agent" and is an example of integrating file output with AI reasoning.
- **Interruptible Demo Agent (`interrupt_agent.py`):** A demonstration agent showing how to use LangGraph’s **`Interrupt`** feature. This agent tries to determine a user's birthdate from conversation. If the birthdate isn’t provided, it issues an interrupt (asks the user for the missing info) and then resumes processing once the user responds. It also shows use of the agent’s **Store** for long-term memory: it will remember the user’s birthdate in a persistent store (if configured) so it doesn’t ask again in future sessions. This is a template for building agents that require multi-turn reasoning with conditional queries to the user.

*(Additional agents or variations may exist in this directory; for example, older prototypes like `coding_agent2` or specialized agents not registered by default. Each file should document the agent’s purpose and usage.)*

## Input/Output and Interface

All agents are structured to receive a **user input** (typically a message string and some metadata) and produce a **final output** (typically an `AIMessage` or a similar structure containing the assistant’s answer). In LangGraph terms, an agent is either:
- a `@entrypoint` function returning a result (like `chatbot` which yields a `messages` list with the response), or 
- a `StateGraph`/`CompiledStateGraph` object (for more complex agents with multiple nodes and steps, like the coding and supervisor agents).

Agents expose a `.name` attribute (set at the end of each module) which is used as the identifier. For example, `chatbot.name = "General Chatbot"`, `coding_agent2.name = "coding_expert"`, etc. The FastAPI service uses these names to route requests to the correct agent. Agents are stored in a registry (`agents` dict in `agent_registry.py`) mapping the name to the agent object and a short description.

**Agent Execution:** At runtime, the service will call an agent by invoking it with a `UserInput` payload. If the agent is a simple function (entrypoint), it is called directly (potentially asynchronously). If it’s a `StateGraph` or similar, the service calls `agent.ainvoke(...)` or `agent.invoke(...)` to get the result. All agents use a consistent input format: a dictionary or Pydantic model containing at least the user’s message, and possibly `thread_id`, `user_id`, `model` override, and an `agent_config` dict for additional parameters. They return either a final `ChatMessage` or (for streaming) an async generator of messages. The specifics of input fields are defined in `schema.UserInput` (see **src/schema** docs). 

**Tools and Dependencies:** Agents can leverage tools from LangChain/LangGraph. In our code:
- The **Coding Expert** binds a `python_repl` tool (which is defined in its module) allowing execution of code. It imports common libraries (`numpy`, `pandas`, `matplotlib`, etc.) so they are available in the REPL’s global scope.
- The **Supervisor Agent** uses tools created by `langgraph_supervisor` such as `create_handoff_tool` to pass tasks to sub-agents. It also imports custom tools `JACSKE_database_search` and `JACSKE_get_full_doc_text` from `agents/tools.py` to query a document database.
- The **Excel Agent** uses a tool `write_to_excel` from `tools/excel_tools.py` to create or update an Excel file with given content. It conditionally uses an OpenWeatherMap tool if an API key is present.
- The **Interrupt Agent** doesn’t use external tools, but it does use the LangGraph **interrupt** mechanism (`langgraph.types.interrupt`) to request additional input from the user mid-run.

All these agents rely on the base LLM model, obtained via `core.get_model(...)`. They typically call `get_model(settings.DEFAULT_MODEL)` at initialization to get an `LLM` instance (or they retrieve a model per request via the `config` passed in, which may contain a model override). The LangGraph framework takes care of streaming token handling and state management under the hood.

## Module Contribution and Orchestration

Each agent module contributes to the overall system by handling a specific class of user requests:
- Simpler queries are handled by the **General Chatbot** agent.
- Programming or data-related queries can be directed to the **Coding Expert** agent.
- Complex documentation/coding queries are managed by the **Supervisor** agent orchestrating multiple skills.
- Specialized workflows (like the Excel output example or the Interrupt demo) showcase how the platform can be extended with domain-specific logic.

The FastAPI service orchestrates these agents by picking the right agent based on the URL path (`/{agent}/invoke`). Internally, the `get_agent()` lookup in `agent_registry.py` retrieves the appropriate agent object for a given key. This decoupling means you can add or remove agents without altering the service logic – just update the registry and ensure the agent has a unique name.

**State and Memory:** Agents can maintain state across turns. When the service calls an agent, it sets up memory (if configured) so that:
- The agent’s short-term memory (conversation so far in the current thread) is passed in as `previous` state or via LangGraph’s mechanisms. For instance, the `chatbot` entrypoint function receives a `previous` messages list which it prepends to the current input.
- The agent’s long-term memory store (if using Postgres or other) is accessible via LangGraph’s `store` object. For example, the **Interrupt Agent** uses `store` to save and retrieve the birthdate across separate runs. The `store` is injected by the service at startup for each agent.
- Agents can thus have persistent knowledge or context beyond a single conversation. This is especially useful for maintaining user profiles or referencing prior work.

## How to Add a New Agent

One of the strengths of this platform is the ease of adding custom agents. To add a new agent:

1. **Create the Agent Module:** Write a new Python file in `src/agents/`, e.g., `my_agent.py`. Define your agent logic using LangGraph. This could be as simple as a function with `@entrypoint` (for a linear single-step agent) or a complex `StateGraph` for multi-step workflows. Set a human-readable `name` attribute on your agent object/function.
2. **Register the Agent:** Open `src/agents/agent_registry.py`. Import your new agent at the top, then add an entry to the `agents` dictionary. The key should be `my_agent.name` and the value an `Agent` dataclass with a short description and the agent object. If you want this agent to be the default, update the `DEFAULT_AGENT` constant to your agent’s name.
3. **Expose via API:** No further changes are needed for the API – by registering in `agents`, the `/info` endpoint will list your agent and clients can invoke it by its name. If your agent requires special handling in the UI (for example, a different welcome message or UI components), update `src/streamlit_app.py` accordingly (e.g., add a condition for `if agent == "my_agent": ...` to customize the greeting or inputs).
4. **(Optional) Tools:** If your agent uses new tools, implement them either in the same file or in a new module (e.g., under `src/agents/tools_myagent.py` or in `src/tools/`). Use the `@tool` decorator from LangChain to wrap tool functions, and give each tool a clear `name` and `description`. In your agent logic, bind these tools into the agent (for LangGraph agents, pass the tools list to `create_react_agent` or similar factory).
5. **Document:** Add a brief description of your agent in this AGENT.md file (so future developers or AI copilots understand its purpose). Include any special instructions or requirements (e.g., needs certain environment variables, or make sure to add keys to .env.example).
6. **Test:** Create tests for your agent if possible. At minimum, ensure that running your agent in isolation produces the expected output for a sample input, and that it doesn’t break the service.

By following these steps, your custom agent will seamlessly integrate into the Agent Platform. The system is designed so that coding agents can read this documentation and adjust their behavior accordingly, making the addition of new capabilities a smooth process.
