import importlib.util
import pathlib
import sys
import types

from starlette.datastructures import Headers, QueryParams


def _load_module(module_name: str, source_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, pathlib.Path(source_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    sys.modules[module_name] = module
    return module


service_package = types.ModuleType("service")
service_package.__path__ = []  # type: ignore[attr-defined]
sys.modules.setdefault("service", service_package)

storage_module = _load_module("service.storage", "src/service/storage.py")
catalog_module = _load_module("service.catalog_postgres", "src/service/catalog_postgres.py")
service_package.storage = storage_module
service_package.catalog_postgres = catalog_module

_FILES_ROUTER_SPEC = importlib.util.spec_from_file_location(
    "service.files_router", pathlib.Path("src/service/files_router.py")
)
files_router = importlib.util.module_from_spec(_FILES_ROUTER_SPEC)
assert _FILES_ROUTER_SPEC and _FILES_ROUTER_SPEC.loader
_FILES_ROUTER_SPEC.loader.exec_module(files_router)  # type: ignore[union-attr]


def make_request(query: dict[str, str] | None = None, headers: dict[str, str] | None = None):
    request = types.SimpleNamespace()
    request.query_params = QueryParams(query or {})
    request.headers = Headers(headers or {})
    return request


def test_build_langfuse_runtime_from_request():
    request = make_request(
        {"trace_id": "query-trace", "session_id": "query-session"},
        {
            files_router.LANGFUSE_TRACE_HEADER: "header-trace",
            files_router.LANGFUSE_SESSION_HEADER: "header-session",
            files_router.LANGFUSE_RUN_HEADER: "header-run",
        },
    )
    runtime = files_router._build_langfuse_runtime(
        request, user_id="user-123", thread_id="thread-456"
    )
    assert runtime == {
        "user_id": "user-123",
        "thread_id": "thread-456",
        "trace_id": "query-trace",
        "session_id": "query-session",
        "run_id": "header-run",
    }


def test_emit_file_event_includes_runtime(monkeypatch):
    captured: dict[str, object] = {}

    def fake_emit(name, runtime, metadata=None):
        captured["name"] = name
        captured["runtime"] = runtime
        captured["metadata"] = metadata

    monkeypatch.setattr(files_router, "emit_runtime_event", fake_emit)
    runtime = {"user_id": "user-123", "thread_id": "thread-456"}

    files_router._emit_file_event(
        "service.files.test",
        runtime,
        filename="test.txt",
        file_size=1024,
    )

    assert captured["name"] == "service.files.test"
    assert captured["runtime"] == runtime
    assert captured["metadata"]["filename"] == "test.txt"
    assert captured["metadata"]["file_size"] == 1024
    assert captured["metadata"]["user_id"] == "user-123"
    assert captured["metadata"]["thread_id"] == "thread-456"
