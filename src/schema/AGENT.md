# Schema Module (src/schema)

The schema module defines the data models and types used by the Agent Platform for communication between components (client, service, agents). It primarily uses **Pydantic** models and some standard Python typing to ensure that requests and responses have a well-defined structure. Having a clear schema is crucial for both runtime validation and for AI coding assistants to understand the expected inputs/outputs of the system.

## Purpose

In summary, `src/schema` provides:
- **Request Models:** Data structures for inputs to the agent service (e.g., what an invocation or stream request looks like).
- **Response Models:** Data structures for outputs from agents or the service (e.g., chat messages, metadata lists).
- **Shared Types:** Any enumerations or constants that need to be consistent across the system (e.g., supported model names).
- **Documentation and Validation:** By defining these models, FastAPI automatically generates API docs and performs validation on incoming data. It also ensures that as developers (or AI agents) work with the code, they have a single source of truth for the shape of data structures.

## Data Models Overview

Key classes and their roles:

- **AgentInfo (BaseModel):** Represents basic info about an agent. Fields:
  - `key` (str): The agent’s identifier (used in URLs).
  - `description` (str): A human-friendly description of the agent.
  This is used in the `/info` endpoint to advertise which agents are available.

- **ServiceMetadata (BaseModel):** Metadata about the running service. Fields:
  - `agents` (List[AgentInfo]): List of all available agents (name and description).
  - `models` (List[AllModelEnum]): List of all supported model names (enums) that the service can use.
  - `default_agent` (str): The default agent’s key.
  - `default_model` (AllModelEnum): The default model in use.
  The `/info` endpoint returns this structure, allowing clients to dynamically adjust to the service’s capabilities.

- **UserInput (BaseModel):** The primary input payload when invoking an agent. Fields:
  - `message` (str): The user’s message or question for the agent.
  - `model` (AllModelEnum | None): (Optional) specify which model the agent should use. If omitted, the service will use `settings.DEFAULT_MODEL`. This allows overriding the LLM per request.
  - `thread_id` (str | None): An identifier for the conversation thread. If provided, the agent can fetch prior context for that thread (enabling multi-turn conversations). If not provided, a new thread is assumed.
  - `user_id` (str | None): (Optional) user identifier for persistence. The service usually ignores any client-sent user_id (it derives actual user_id from auth), so this is typically not set by clients.
  - `agent_config` (dict[str, Any]): A catch-all for any agent-specific parameters. Agents can define custom config keys (for example, a hypothetical agent might accept `{"temperature": 0.9}` or domain-specific flags). By default it’s an empty dict.
  All fields except message have defaults, so a minimal valid UserInput is just `{"message": "Hello"}`. The service will fill in thread_id, etc., if needed.

- **StreamInput (UserInput):** Inherits all fields from UserInput and adds one more:
  - `stream_tokens` (bool): If true (default), the service streams partial LLM tokens as they are generated. If false, the service may still stream but only complete messages or tool outputs. This gives fine control over streaming granularity. In practice, this is usually left true for real-time token streaming. 
  The `StreamInput` is used for the `/stream` endpoint. It ensures the request structure is the same as UserInput, just with this extra flag.

- **ChatMessage (BaseModel):** Represents a message in the chat (either user, AI, tool, or system). Fields:
  - `type` (Literal["human", "ai", "tool", "custom"]): The role or type of message. "human" = user message, "ai" = assistant/agent response, "tool" = a tool’s action or output, "custom" = other system messages or data messages.
  - `content` (str): The text content of the message. If this message is from the AI, it's the answer text. If from a tool, it might be a description of the action or result.
  - `tool_calls` (List[ToolCall]): Any tool invocations that occurred within this message. (For example, if an AI message says it’s using a tool, the details are captured here.)
  - `tool_call_id` (str | None): If this message is a response to a tool’s output, this might link back to the tool call.
  - `run_id` (str | None): The unique ID of the agent run this message is part of.
  - `response_metadata` (dict): Additional metadata about the response (e.g., token usage, model info, etc.).
  - `custom_data` (dict): Any custom payload the agent might include (for example, some agents might include structured data here).
  
  The ChatMessage model is used in responses. For non-stream invoke, the service returns a single ChatMessage of type "ai". For streams, it may send multiple: e.g., interim ones of type "tool" or "custom" (progress updates), and final "ai" message.

  There are also convenience methods:
  - `pretty_repr()` and `pretty_print()`, which format the message nicely for console output (with a title line and the content). These are used in the AgentClient to print responses clearly.

- **ToolCall (TypedDict):** A structure for logging a tool invocation within a message. Fields:
  - `name` (str): Tool name.
  - `args` (dict[str, Any]): Arguments used for the tool call.
  - `id` (str | None): An optional ID for the tool call (could correlate with `tool_call_id` in a later message).
  - `type`: (literal "tool_call", optional) – If included, likely just a marker to differentiate this dict as a tool call event.
  ToolCall is not a Pydantic model but a TypedDict (used for structure). It’s serialized as part of ChatMessage to describe what actions the agent took. 

- **Feedback (BaseModel):** Represents a feedback entry for a particular run. Fields:
  - `run_id` (str): The ID of the agent run this feedback is about.
  - `key` (str): A key identifying the kind of feedback (e.g., "human-feedback-rating" or "model-evaluation-score").
  - `score` (float): A numerical score or rating. Could be 1-5 stars, or a probability, etc., depending on context.
  - `kwargs` (dict): Additional metadata about the feedback (e.g., comments).
  This could be used by the `/feedback` endpoint to log user feedback into LangSmith or another tracking system.

- **FeedbackResponse (BaseModel):** A simple acknowledgment model for feedback submissions. It has one field `status` which is "success" if the feedback was recorded. The service might return this after processing a Feedback input.

- **ChatHistoryInput (BaseModel):** A request model to retrieve a conversation history. Field:
  - `thread_id` (str): The thread whose history is requested.
  If the service implements a route like `POST /history` or similar, this model would be used to validate the request.

- **ChatHistory (BaseModel):** The response model containing past messages. Field:
  - `messages` (List[ChatMessage]): The list of messages (in order) that have occurred in the thread.
  This would be returned by a history retrieval endpoint or could be used internally to pass conversation context.

In addition to classes, `schema/models.py` defines enumerations:
- **AllModelEnum:** A union of all model enum classes (OpenAI, AzureOpenAI, Groq, Ollama, etc.). This is used as the type for model fields so that any supported model can be referenced. 
- Enums like `OpenAIModelName`, `AzureOpenAIModelName`, `GroqModelName`, `OllamaModelName`, `FakeModelName` etc. These are typically Python `enum.Enum` or `StrEnum` that list the allowed model identifiers. For example, `OpenAIModelName` might have values like "GPT_4", "GPT_3_5" etc., corresponding to the underlying API model names.
- These enums are important: they ensure that if a client specifies `model: "GPT_4"`, it matches one of the known values. If not, Pydantic will throw a validation error (422 response), preventing unsupported model usage.
- The `AllModelEnum` is constructed to include all possible models (the code likely uses `enum.Flag` or a custom Union approach). In our context, `AllModelEnum` is used in `ServiceMetadata.models` and as the type for `UserInput.model` field.

By structuring model choices as enums, it also helps any AI developer working with the code to see what models are expected or available.

## How Schema Is Used

- **FastAPI Integration:** Each Pydantic model is used in path or request/response definitions. For example, in `service.py`, the function signature `invoke(user_input: UserInput, ...) -> ChatMessage` tells FastAPI to:
  - Parse the JSON body into `UserInput` (validate required fields).
  - Serialize the output `ChatMessage` into JSON when returning.
- **AgentClient:** The client library uses these models too. For instance:
  - `ServiceMetadata.model_validate(response.json())` to parse `/info` response into a Python object.
  - It constructs a `UserInput` when sending a request (`request = UserInput(message=..., ...)` in AgentClient.invoke). Actually, in the snippet, it directly uses Pydantic by calling `json=request.model_dump()` to send it over HTTP.
  - It receives a response and does `ChatMessage.model_validate(response.json())` to parse it into a `ChatMessage` object.
  This means the client and server share these models, which is beneficial for consistency.

- **Agents usage:** Agents themselves might not use Pydantic models extensively (they often work with internal LangGraph message objects), but they do interact with these schema definitions indirectly:
  - The `agents/agent_registry.py` uses `AgentInfo` to construct the list for `/info`.
  - Tools or outputs might yield structures that match `ToolCall` or update `ChatMessage.custom_data`.
  - If an agent expects some structured input in `agent_config`, that structure could be defined here (though currently `agent_config` is just a `dict[str, Any]` – dynamic).

## Input/Output Expectations

By having a defined schema:
- **Inputs:** The service will only call an agent with validated, structured data (for example, `user_input.message` is guaranteed to be a string with content, not null).
- **Outputs:** The agent’s response is converted to a `ChatMessage` which ensures it has a type and content. Agents should populate the `content` with their answer, and can optionally include details of tool usage. For instance, if an agent used a tool to retrieve info, it might include that in `tool_calls` so the client or UI can present it or log it.

For developers and AI coding assistants, this schema acts like a contract:
- Knowing the `ChatMessage` format, for example, an AI assistant can decide to put certain data in `custom_data` vs `content` if it’s implementing a new feature (like returning a chart as a data URI, which the UI might detect).
- The `Feedback` model indicates how feedback is expected to be sent, if implementing an automated evaluation.

## Example

To illustrate, consider a conversation:
User says: "What’s the weather in Tokyo?"
The service will create `UserInput(message="What's the weather in Tokyo?", thread_id=XYZ)`.
After agent processing, suppose the agent uses a Weather tool and then answers:
The final `ChatMessage` might look like:
```json
{
  "type": "ai",
  "content": "The weather in Tokyo is sunny and 25°C.",
  "tool_calls": [
    {"name": "WeatherAPI", "args": {"location": "Tokyo"}, "id": "call_1"}
  ],
  "tool_call_id": null,
  "run_id": "123e4567-e89b-12d3-a456-426614174000",
  "response_metadata": {"tokens_used": 50},
  "custom_data": {}
}
