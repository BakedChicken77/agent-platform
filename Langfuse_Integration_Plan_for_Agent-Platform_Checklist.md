### ✅ **Step 1 Checklist Review**

#### 1. **Create `src/core/langfuse.py` helper**

**✔ Done.**

* Implemented a lazily cached `get_langfuse_client()`, `get_langfuse_handler()`, and `create_span()` factory.
* Handles optional Langfuse SDK gracefully (with local imports, error handling).
* Uses `settings.LANGFUSE_*` and guards against missing secrets.
* Uses `nullcontext` when tracing is disabled (ensures app start safety).

#### 2. **Extend `Settings` and `.env.example`**

**✔ Done.**

* `Settings` model updated with:

  * `LANGFUSE_TRACING`, `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_CLIENT_PUBLIC_KEY`, `LANGFUSE_ENVIRONMENT`, `LANGFUSE_SAMPLE_RATE`
* `.env.example` extended accordingly.
* Types and defaults handled correctly, e.g., `SecretStr` and nullable strings/floats.

#### 3. **Update `compose.yaml` + Dockerfiles**

**✔ Done.**

* `compose.yaml`:

  * Added `LANGFUSE_*` to both `build.args` and `environment` for `agent_service` and `streamlit_app`.
* `Dockerfile.app` and `Dockerfile.service`:

  * Accept all `LANGFUSE_*` as `ARG`s and promote to `ENV`.

#### 4. **Document in `README.md`**

**✔ Done.**

* Added a clear “Enabling Langfuse Tracing” section:

  * Includes setup steps, `.env` example, restart instructions, and `/health` check usage.

#### 5. **Guard application when tracing is off**

**✔ Done.**

* `get_langfuse_client()` returns `None` if tracing is disabled or SDK not installed.
* `create_span()` falls back to `nullcontext()`.
* `get_langfuse_handler()` returns `None` if SDK or keys are missing.
* `/health` route defends against failed SDK loading or client errors.

#### 6. **Basic regression test coverage**

**✔ Done.**

* Added `tests/core/test_langfuse.py` with 3 robust test cases:

  * No SDK / no tracing
  * Simulated SDK failure
  * Dummy SDK with env variable overrides
* Uses `monkeypatch`, `sys.modules`, and `pytest` fixtures properly.
* Clears internal caches to ensure clean test environments.

