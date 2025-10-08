import importlib.util
import io
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "image/png",
    "image/jpeg",
    "image/webp",
    "application/x-python-code",
}


class _Settings:
    ALLOWED_MIME_TYPES = ALLOWED_MIME_TYPES
    MAX_UPLOAD_MB = 100
    AUTO_INGEST_UPLOADS = False


stub_settings_module = types.ModuleType("core.settings")
stub_settings_module.settings = _Settings()
sys.modules["core.settings"] = stub_settings_module

stub_core_module = types.ModuleType("core")
stub_core_module.settings = stub_settings_module.settings
sys.modules["core"] = stub_core_module

stub_service_module = types.ModuleType("service")
stub_service_module.__path__ = []  # mark as package
sys.modules["service"] = stub_service_module

stub_catalog_module = types.ModuleType("service.catalog_postgres")
stub_catalog_module.get_by_sha_and_name = lambda *_, **__: None
stub_catalog_module.save_metadata = lambda *_, **__: None
stub_catalog_module.list_metadata = lambda *_, **__: []
stub_catalog_module.get_metadata_by_id = lambda *_, **__: None
stub_catalog_module.delete_metadata = lambda *_, **__: None
sys.modules["service.catalog_postgres"] = stub_catalog_module
stub_service_module.catalog_postgres = stub_catalog_module


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


storage = _load_module("src/service/storage.py", "service.storage")
stub_service_module.storage = storage

files_router = _load_module("src/service/files_router.py", "service.files_router")

sniff_mime_from_upload = files_router.sniff_mime_from_upload
is_allowed = storage.is_allowed


@dataclass
class _StubUpload:
    filename: str
    file: SpooledTemporaryFile
    content_type: str | None = None


def _upload_from_bytes(data: bytes, filename: str) -> _StubUpload:
    file_obj = SpooledTemporaryFile()
    file_obj.write(data)
    file_obj.seek(0)
    return _StubUpload(filename=filename, file=file_obj)


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("doc.txt", "hello")
    return buffer.getvalue()


def test_generic_zip_rejected() -> None:
    upload = _upload_from_bytes(_zip_bytes(), "archive.zip")
    mime = sniff_mime_from_upload(upload)
    assert mime == "application/zip"
    assert not is_allowed(mime)


def test_excel_zip_allowed() -> None:
    upload = _upload_from_bytes(_zip_bytes(), "sheet.xlsx")
    mime = sniff_mime_from_upload(upload)
    assert (
        mime
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert is_allowed(mime)
