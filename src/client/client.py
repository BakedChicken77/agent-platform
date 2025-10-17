# src/client/client.py

import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import IO, Any

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)


class AgentClientError(Exception):
    pass


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
        access_token: str | None = None,  # OAuth2 bearer token (JWT)
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
            agent (str): The name of the default agent to use.
            timeout (float, optional): The timeout for requests.
            get_info (bool, optional): Whether to fetch agent information on init.
                Default: True
            access_token (str, optional): Bearer token obtained via Azure AD OAuth2.
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.access_token = access_token  # store OAuth2 token
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    # -------- helper --------
    def set_token(self, token: str | None) -> None:
        """Update the bearer token at runtime (e.g., per-session OAuth2 token)."""
        self.access_token = token
    # ------------------------

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        elif self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def _with_langfuse_headers(
        self,
        *,
        base: dict[str, str] | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, str]:
        headers = dict(base or self._headers)
        if trace_id:
            headers["X-Langfuse-Trace-Id"] = trace_id
        if session_id:
            headers["X-Langfuse-Session-Id"] = session_id
        if run_id:
            headers["X-Langfuse-Run-Id"] = run_id
        return headers

    def _prepare_agent_config(
        self,
        agent_config: dict[str, Any] | None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Merge Langfuse identifiers into ``agent_config`` when provided."""

        extras = {
            key: value
            for key, value in {"trace_id": trace_id, "session_id": session_id}.items()
            if value
        }
        if not extras:
            return agent_config

        config = dict(agent_config or {})
        langfuse_config = dict(config.get("langfuse", {}))
        langfuse_config.update(extras)
        config["langfuse"] = langfuse_config
        return config

    def retrieve_info(self) -> None:
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Identity is derived server-side from the JWT; no client-supplied user_id is sent.
        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            trace_id (str, optional): Existing Langfuse trace identifier to reuse
            session_id (str, optional): Session identifier for Langfuse (defaults to thread_id)

        Returns:
            AnyMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        agent_config = self._prepare_agent_config(
            agent_config, trace_id=trace_id, session_id=session_id
        )
        if agent_config:
            request.agent_config = agent_config
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> ChatMessage:
        """
        Invoke the agent synchronously. Only the final message is returned.

        Identity is derived server-side from the JWT; no client-supplied user_id is sent.
        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            trace_id (str, optional): Existing Langfuse trace identifier to reuse
            session_id (str, optional): Session identifier for Langfuse (defaults to thread_id)

        Returns:
            ChatMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        agent_config = self._prepare_agent_config(
            agent_config, trace_id=trace_id, session_id=session_id
        )
        if agent_config:
            request.agent_config = agent_config
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # Convert the JSON formatted message to a ChatMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # Yield the str token directly
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        Stream the agent's response synchronously.

        Identity is derived server-side from the JWT; no client-supplied user_id is sent.
        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True
            trace_id (str, optional): Existing Langfuse trace identifier to reuse
            session_id (str, optional): Session identifier for Langfuse (defaults to thread_id)

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        agent_config = self._prepare_agent_config(
            agent_config, trace_id=trace_id, session_id=session_id
        )
        if agent_config:
            request.agent_config = agent_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        Stream the agent's response asynchronously.

        Identity is derived server-side from the JWT; no client-supplied user_id is sent.

        Each intermediate message of the agent process is yielded as an AnyMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming modelsas they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True
            trace_id (str, optional): Existing Langfuse trace identifier to reuse
            session_id (str, optional): Session identifier for Langfuse (defaults to thread_id)

        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        agent_config = self._prepare_agent_config(
            agent_config, trace_id=trace_id, session_id=session_id
        )
        if agent_config:
            request.agent_config = agent_config
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        kwargs: dict[str, Any] | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        langfuse_run_id: str | None = None,
    ) -> FeedbackResponse:
        """
        Create a feedback record for a run.

        This is a simple wrapper for the LangSmith create_feedback API, so the
        credentials can be stored and managed in the service rather than the client.
        See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        payload_kwargs: dict[str, Any] = dict(kwargs or {})
        metadata: dict[str, Any] | None = None
        if payload_kwargs.get("metadata") is not None:
            metadata = dict(payload_kwargs["metadata"])
        elif trace_id or session_id:
            metadata = {}

        if metadata is not None:
            if trace_id:
                metadata["langfuse_trace_id"] = trace_id
            if session_id:
                metadata["langfuse_session_id"] = session_id
            payload_kwargs["metadata"] = metadata

        request = Feedback(
            run_id=run_id,
            key=key,
            score=score,
            kwargs=payload_kwargs,
            trace_id=trace_id,
            session_id=session_id,
            langfuse_run_id=langfuse_run_id,
        )
        payload = request.model_dump(exclude_none=True)
        headers = self._with_langfuse_headers(
            trace_id=trace_id,
            session_id=session_id,
            run_id=langfuse_run_id,
        )
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return FeedbackResponse.model_validate(response.json())
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str): Thread ID for identifying a conversation
        """
        request = ChatHistoryInput(thread_id=thread_id)
        try:
            response = httpx.post(
                f"{self.base_url}/history",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

    # ---------- Files API ----------
    def list_files(
        self,
        thread_id: str | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        langfuse_run_id: str | None = None,
    ) -> dict:
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        if trace_id:
            params["trace_id"] = trace_id
        if session_id:
            params["session_id"] = session_id
        if langfuse_run_id:
            params["run_id"] = langfuse_run_id
        headers = self._with_langfuse_headers(
            trace_id=trace_id,
            session_id=session_id,
            run_id=langfuse_run_id,
        )
        try:
            r = httpx.get(
                f"{self.base_url}/files",
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"List files failed: {e}")

    def download_file(
        self,
        file_id: str,
        thread_id: str | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        langfuse_run_id: str | None = None,
    ) -> bytes:
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        if trace_id:
            params["trace_id"] = trace_id
        if session_id:
            params["session_id"] = session_id
        if langfuse_run_id:
            params["run_id"] = langfuse_run_id
        headers = self._with_langfuse_headers(
            trace_id=trace_id,
            session_id=session_id,
            run_id=langfuse_run_id,
        )
        params["download"] = "true"
        try:
            r = httpx.get(
                f"{self.base_url}/files/{file_id}",
                params=params,
                headers=headers,
                timeout=None,
            )
            r.raise_for_status()
            return r.content
        except httpx.HTTPError as e:
            raise AgentClientError(f"Download failed: {e}")

    def delete_file(
        self,
        file_id: str,
        thread_id: str | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        langfuse_run_id: str | None = None,
    ) -> None:
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        if trace_id:
            params["trace_id"] = trace_id
        if session_id:
            params["session_id"] = session_id
        if langfuse_run_id:
            params["run_id"] = langfuse_run_id
        headers = self._with_langfuse_headers(
            trace_id=trace_id,
            session_id=session_id,
            run_id=langfuse_run_id,
        )
        try:
            r = httpx.delete(
                f"{self.base_url}/files/{file_id}",
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Delete failed: {e}")

    def upload_files(
        self,
        files: list[tuple[str, bytes | IO[bytes], str | None]],
        thread_id: str | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        langfuse_run_id: str | None = None,
    ) -> list[dict]:
        """
        Upload multiple files.

        files: list of tuples (field_name, data_or_stream, mime) where field_name is the original filename.
        """
        multipart = []
        for name, data, mime in files:
            multipart.append(("files", (name, data, mime or "application/octet-stream")))
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        if trace_id:
            params["trace_id"] = trace_id
        if session_id:
            params["session_id"] = session_id
        if langfuse_run_id:
            params["run_id"] = langfuse_run_id
        headers = self._with_langfuse_headers(
            trace_id=trace_id,
            session_id=session_id,
            run_id=langfuse_run_id,
        )
        try:
            r = httpx.post(
                f"{self.base_url}/files/upload",
                params=params,
                files=multipart,
                headers=headers,
                timeout=None,
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Upload failed: {e}")

    async def aupload_files(
        self,
        files: list[tuple[str, bytes | IO[bytes], str | None]],
        thread_id: str | None = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        langfuse_run_id: str | None = None,
    ) -> list[dict]:
        multipart = []
        for name, data, mime in files:
            multipart.append(("files", (name, data, mime or "application/octet-stream")))
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        if trace_id:
            params["trace_id"] = trace_id
        if session_id:
            params["session_id"] = session_id
        if langfuse_run_id:
            params["run_id"] = langfuse_run_id
        headers = self._with_langfuse_headers(
            trace_id=trace_id,
            session_id=session_id,
            run_id=langfuse_run_id,
        )
        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(
                    f"{self.base_url}/files/upload",
                    params=params,
                    files=multipart,
                    headers=headers,
                    timeout=None,
                )
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Upload failed: {e}")
