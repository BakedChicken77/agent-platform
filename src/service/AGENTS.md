
# Service Module (`src/service`)

The **Service Module** implements the FastAPI web service that exposes AI agents via RESTful endpoints.  
It connects external users (or client applications) to internal agent logic, managing routing, async execution, streaming, authentication, and persistence (memory and files).

---

## Purpose

The FastAPI service serves as the API **front-end** for the Agent Platform. It allows clients to:

- Retrieve information about available agents and models.
- Invoke a specific agent with a given input and receive the final response.
- Stream an agent’s response in real time.
- Upload files for agents to use (e.g., reference documents or data files).
- *(Optionally)* Submit feedback on agent responses.

The service ensures that each request is properly authenticated (if enabled) and that the database and filesystem are correctly initialized and utilized.

---

## API Endpoints

### **GET `/info`**

Returns metadata about the service, including available agents and models.  
The response matches the `ServiceMetadata` model (see **schema** documentation).

Example response:

```json
{
  "agents": [
    {"key": "General Chatbot", "description": "A simple chatbot."},
    {"key": "coding_expert", "description": "A Python Coding Agent"}
  ],
  "models": ["GPT_4", "GPT_3_5"],
  "default_agent": "General Chatbot",
  "default_model": "GPT_4"
}
````

Clients typically call this on startup to discover which agents and models are available.

---

### **POST `/{agent_id}/invoke`**

Invokes an agent with a single input and returns the final response.

* **`agent_id`** — The agent’s name or key (as listed in `/info`).
* **Request Body:** Matches the `UserInput` model.

  ```json
  {
    "message": "Generate a report summary.",
    "thread_id": "abc123",
    "model": "GPT_4"
  }
  ```
* **Response:** Returns a `ChatMessage` object (includes type, content, and optional tool call traces).

---

### **POST `/{agent_id}/stream`**

Similar to `/invoke`, but streams the response in real time.

* The connection remains open, and data is sent as a series of `data:` lines (Server-Sent Events).
* Each chunk is a partial JSON fragment, typically a `ChatMessage` or token output.
* When complete, the stream ends with `data: [DONE]`.

**Request Body:** Uses the `StreamInput` model (inherits from `UserInput`, adds `stream_tokens: true`).

Clients can set `stream_tokens: false` to receive chunked messages rather than tokens.

---

### **POST `/files/upload`**

Uploads one or multiple files for agents to use.

* **Form:** `multipart/form-data`
* **Fields:** `files` (one or more), optional `thread_id`
* **Response:** List of results per file (each includes success status and file `id`).

**File Storage:**

* Default path: `uploads/{user_id}/{thread_id}/`
* Postgres variant: `catalog_postgres` helper indexes file metadata for retrieval.
* Only allowed MIME types (e.g., text, PDF, image).
* File size limit: `MAX_UPLOAD_MB`.

**Additional Routes:**

* `GET /files` — List files (`ListFilesResponse`).
* `GET /files/{file_id}` — Download file contents.

---

### **POST `/feedback`**

*(Optional)* Accepts user feedback on agent responses.

* Uses the `Feedback` model (`run_id`, `score`, `key`, etc.).
* Could log feedback to analytics systems like LangSmith or Langfuse.

---

### **Health and Docs**

* **Health Check:** `/info` doubles as a health endpoint (returns 200 when initialized).
* **Redoc Docs:** Accessible at `http://localhost:8080/redoc` (FastAPI auto-docs).

---

## Service Lifecycle and Execution Flow

### **Startup Sequence (`service.py`)**

1. **Load Environment Variables**

   * `load_dotenv()` + `get_settings()`

2. **Setup Middleware**

   * **AuthMiddleware:** Verifies JWTs via Azure AD. Populates `request.state.user` or rejects requests.
   * **LoggingMiddleware:** Logs requests and latency to `"http"` logger.
   * **CORS Middleware:** Allows all origins (development mode). Restrict in production.

3. **Initialize Persistence (Lifespan Context)**

   * Calls `initialize_database()` and `initialize_store()` from `memory/__init__.py`.
   * Connects to `SQLite`, `Postgres`, or `Mongo` depending on `settings.DATABASE_TYPE`.
   * Attaches memory components to each agent:

     ```python
     agent.checkpointer = saver  # short-term memory
     agent.store = store         # long-term memory
     ```
   * Preserves conversation history and vector knowledge across requests.

4. **Include Routers**

   * Main API routes: `/info`, `/invoke`, `/stream`.
   * File routes: `/files` (from `service/files_router.py`).

---

### **Request Flow Example (`POST /{agent}/invoke`)**

1. FastAPI validates JWT and populates `request.state.user`.
2. Calls endpoint:

   ```python
   async def invoke(user_input: UserInput, request: Request, agent_id: str = DEFAULT_AGENT)
   ```
3. Retrieves the agent via `get_agent(agent_id)`.
4. Prepares runtime config using `_handle_input`:

   * Generates new `run_id` and `thread_id`.
   * Adds `user_id` (from JWT claims).
   * Merges `agent_config`.
   * Handles interrupts: resumes paused runs via `Command(resume=...)` if applicable.
5. Invokes the agent:

   ```python
   result = await agent.ainvoke(**kwargs)
   ```
6. Returns the result as a `ChatMessage` JSON object.

### **Streaming (`POST /{agent}/stream`)**

* Iterates over agent’s async generator.
* Sends `ChatMessage` chunks via `StreamingResponse(media_type="text/event-stream")`.
* Allows live token-by-token or message streaming.

---

## Authentication & Authorization

* Controlled via **AuthMiddleware** from `src/auth`.
* Requires **Bearer JWT** (Azure AD access token) unless `AUTH_ENABLED=False`.

### **Key Behaviors:**

* Validates token against Azure JWKS and `api://<AZURE_AD_API_CLIENT_ID>` audience.
* Stores claims in `request.state.user`.
* Blocks invalid/expired tokens (401 Unauthorized).
* Development mode (`AUTH_ENABLED=False`) injects:

  ```python
  request.state.user = {"username": "devuser"}
  ```

Role-based access is not implemented but can be extended via JWT `roles` claims.

---

## Memory Integration

### **Checkpointer (Short-Term Memory)**

* Saves each conversation turn (via `AsyncSqliteSaver` or `AsyncPostgresSaver`).
* Restores history by `user_id` and `thread_id`.

### **Store (Long-Term Memory)**

* Vector or key-value store for persistent knowledge.
* Postgres uses `pgvector`; SQLite uses in-memory or local alternatives.
* Tracks embeddings, documents, or file metadata.

### **Scope**

* **User:** Derived from JWT claim (isolates data between users).
* **Thread:** Conversation session (enables contextual continuity).

---

## File Handling

* **Uploads:** Managed via `/files/upload`.
* Each file:

  * Assigned a unique `file_id` (UUID).
  * Written atomically to disk (avoiding partial writes).
  * Metadata (filename, MIME, size) stored for retrieval.
* **Allowed Types:** Configured in `settings.ALLOWED_MIME_TYPES`.
* **Listing/Download:** Provided via `/files` and `/files/{file_id}`.



