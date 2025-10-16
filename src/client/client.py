# src/client/client.py

import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, IO

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
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
        trace_id: str | None = None,
        session_id: str | None = None,
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
        self.trace_id = trace_id
        self.session_id = session_id
        self._trace_url: str | None = None
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
        if self.trace_id:
            headers["X-Langfuse-Trace-Id"] = self.trace_id
        if self.session_id:
            headers["X-Langfuse-Session-Id"] = self.session_id
        return headers

    @property
    def trace_url(self) -> str | None:
        """Return the latest Langfuse trace URL, if available."""

        return self._trace_url

    def _resolve_trace_context(
        self, trace_id: str | None, session_id: str | None
    ) -> tuple[str | None, str | None]:
        resolved_trace_id = trace_id or self.trace_id
        resolved_session_id = session_id or self.session_id
        return resolved_trace_id, resolved_session_id

    def _merge_agent_config(
        self,
        agent_config: dict[str, Any] | None,
        trace_id: str | None,
        session_id: str | None,
    ) -> dict[str, Any] | None:
        if agent_config is None and not trace_id and not session_id:
            return agent_config

        merged: dict[str, Any] = {}
        if agent_config:
            merged.update(agent_config)
        if trace_id and "trace_id" not in merged:
            merged["trace_id"] = trace_id
        if session_id and "session_id" not in merged:
            merged["session_id"] = session_id
        return merged

    def _update_trace_context(self, *, headers: httpx.Headers | None = None, payload: dict[str, Any] | None = None) -> None:
        trace_id: str | None = None
        trace_url: str | None = None
        session_id: str | None = None

        if headers:
            trace_id = (
                headers.get("x-langfuse-trace-id")
                or headers.get("x-trace-id")
                or headers.get("X-Langfuse-Trace-Id")
            )
            trace_url = (
                headers.get("x-langfuse-trace-url")
                or headers.get("x-trace-url")
                or headers.get("X-Langfuse-Trace-Url")
            )
            session_id = (
                headers.get("x-langfuse-session-id")
                or headers.get("x-session-id")
                or headers.get("X-Langfuse-Session-Id")
            )

        if payload:
            trace_id = payload.get("trace_id") or trace_id
            trace_url = payload.get("trace_url") or trace_url
            payload_metadata = payload.get("response_metadata")
            if isinstance(payload_metadata, dict):
                trace_id = payload_metadata.get("trace_id") or trace_id
                trace_url = payload_metadata.get("trace_url") or trace_url
            content = payload.get("content")
            if isinstance(content, dict):
                content_metadata = content.get("response_metadata")
                if isinstance(content_metadata, dict):
                    trace_id = content_metadata.get("trace_id") or trace_id
                    trace_url = content_metadata.get("trace_url") or trace_url
            session_id = payload.get("session_id") or session_id

        if trace_id:
            self.trace_id = trace_id
        if session_id:
            self.session_id = session_id
        if trace_url:
            self._trace_url = trace_url

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

        Returns:
            AnyMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        resolved_trace_id, resolved_session_id = self._resolve_trace_context(trace_id, session_id)

        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        merged_config = self._merge_agent_config(agent_config, resolved_trace_id, resolved_session_id)
        if merged_config:
            request.agent_config = merged_config
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

        payload = response.json()
        self._update_trace_context(headers=response.headers, payload=payload)
        return ChatMessage.model_validate(payload)

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
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

        Returns:
            ChatMessage: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        resolved_trace_id, resolved_session_id = self._resolve_trace_context(trace_id, session_id)

        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        merged_config = self._merge_agent_config(agent_config, resolved_trace_id, resolved_session_id)
        if merged_config:
            request.agent_config = merged_config
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

        payload = response.json()
        self._update_trace_context(headers=response.headers, payload=payload)
        return ChatMessage.model_validate(payload)

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
                        message_payload = parsed["content"]
                        self._update_trace_context(payload=parsed)
                        return ChatMessage.model_validate(message_payload)
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

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        resolved_trace_id, resolved_session_id = self._resolve_trace_context(trace_id, session_id)

        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        merged_config = self._merge_agent_config(agent_config, resolved_trace_id, resolved_session_id)
        if merged_config:
            request.agent_config = merged_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                self._update_trace_context(headers=response.headers)
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

        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        resolved_trace_id, resolved_session_id = self._resolve_trace_context(trace_id, session_id)

        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        merged_config = self._merge_agent_config(agent_config, resolved_trace_id, resolved_session_id)
        if merged_config:
            request.agent_config = merged_config
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
                    self._update_trace_context(headers=response.headers)
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
        trace_id: str | None = None,
        trace_url: str | None = None,
    ) -> None:
        """
        Create a feedback record for a run.

        This is a simple wrapper for the LangSmith create_feedback API, so the
        credentials can be stored and managed in the service rather than the client.
        See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        payload_kwargs = kwargs or {}
        request = Feedback(
            run_id=run_id,
            key=key,
            score=score,
            kwargs=payload_kwargs,
            trace_id=trace_id,
            trace_url=trace_url,
        )
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(exclude_none=True),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
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
    def list_files(self, thread_id: str | None = None) -> dict:
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        try:
            r = httpx.get(f"{self.base_url}/files", params=params, headers=self._headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"List files failed: {e}")

    def download_file(self, file_id: str, thread_id: str | None = None) -> bytes:
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        params["download"] = "true"
        try:
            r = httpx.get(f"{self.base_url}/files/{file_id}", params=params, headers=self._headers, timeout=None)
            r.raise_for_status()
            return r.content
        except httpx.HTTPError as e:
            raise AgentClientError(f"Download failed: {e}")

    def delete_file(self, file_id: str, thread_id: str | None = None) -> None:
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        try:
            r = httpx.delete(f"{self.base_url}/files/{file_id}", params=params, headers=self._headers, timeout=self.timeout)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Delete failed: {e}")

    def upload_files(self, files: list[tuple[str, bytes | IO[bytes], str | None]], thread_id: str | None = None) -> list[dict]:
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
        try:
            r = httpx.post(
                f"{self.base_url}/files/upload",
                params=params,
                files=multipart,
                headers=self._headers,
                timeout=None,
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Upload failed: {e}")

    async def aupload_files(self, files: list[tuple[str, bytes | IO[bytes], str | None]], thread_id: str | None = None) -> list[dict]:
        multipart = []
        for name, data, mime in files:
            multipart.append(("files", (name, data, mime or "application/octet-stream")))
        params = {}
        if thread_id:
            params["thread_id"] = thread_id
        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(
                    f"{self.base_url}/files/upload",
                    params=params,
                    files=multipart,
                    headers=self._headers,
                    timeout=None,
                )
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Upload failed: {e}")
