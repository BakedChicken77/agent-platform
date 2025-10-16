## src/service/files_router.py

from __future__ import annotations

import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any

import filetype
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from langfuse import Langfuse  # type: ignore[import-untyped]

from core import settings
from schema.files import FileMeta, ListFilesResponse, UploadResult
from service import catalog_postgres as catalog
from service.storage import (
    build_user_dir,
    is_allowed,
    sha256_streaming,
    write_stream_atomic_with_limit,
)

router = APIRouter(tags=["files"])

SNIFF_BYTES = 8192  # read a small header; safe for large files

logger = logging.getLogger(__name__)

_langfuse_client: Langfuse | None = None
_langfuse_client_failed = False


def _get_langfuse_client() -> Langfuse | None:
    global _langfuse_client, _langfuse_client_failed

    if not settings.LANGFUSE_TRACING or _langfuse_client_failed:
        return None

    if _langfuse_client is None:
        try:
            _langfuse_client = Langfuse()
        except Exception as exc:  # pragma: no cover - defensive guard
            _langfuse_client_failed = True
            logger.warning("Failed to initialize Langfuse client for file events: %s", exc)
            return None

    return _langfuse_client


def _record_file_event(
    *,
    name: str,
    user_id: str,
    thread_id: str | None,
    metadata: dict[str, Any],
) -> None:
    client = _get_langfuse_client()
    if not client:
        return

    payload = {"user_id": user_id, "thread_id": thread_id}
    payload.update(metadata)

    try:
        event = client.event(name=name, metadata=payload)
        event.end()
    except Exception as exc:  # pragma: no cover - telemetry best effort
        logger.debug("Skipping Langfuse event '%s' due to error: %s", name, exc)


def sniff_mime_from_upload(uf) -> str:
    """
    Determine MIME type from file signature first, then fall back to filename and client header.
    Resets the file pointer back to 0 so other readers (hashing/writes) work normally.
    """
    # read a small header
    head = uf.file.read(SNIFF_BYTES)
    # try signature-based detection
    kind = filetype.guess(head)
    # reset pointer so later code can re-read the stream
    uf.file.seek(0)

    ext_mime = mimetypes.guess_type(uf.filename or "")[0]
    fallback_mime = uf.content_type or "application/octet-stream"

    if kind and kind.mime:
        sniffed = kind.mime
        # Prefer the extension-derived MIME for OpenXML formats that look like ZIP
        if (
            sniffed in {"application/zip", "application/octet-stream"}
            and ext_mime
            and ext_mime.startswith("application/vnd.openxmlformats-officedocument")
        ):
            return ext_mime
        return sniffed

    if ext_mime:
        return ext_mime

    # final fallback to client-provided content_type (untrusted)
    return fallback_mime


def _get_user_id_from_claims(claims: dict) -> str:
    """
    Extract a stable per-user identifier from JWT claims.
    We deliberately ignore tenant scoping here to favor (user_id, thread_id) storage.
    """
    return str(
        claims.get("oid")
        or claims.get("sub")
        or claims.get("preferred_username")
        or "unknown"
    )


@router.post("/upload", response_model=list[UploadResult])
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    thread_id: str | None = None,
):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = _get_user_id_from_claims(request.state.user)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Use (user_id, thread_id) for storage partitioning; do not use tenant_id.
    root = build_user_dir(user_id=user_id, thread_id=thread_id, tenant_id=None)
    results: list[UploadResult] = []
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    for uf in files:
        event_file_meta: dict[str, Any] = {
            "original_name": uf.filename,
            "content_type": uf.content_type,
        }
        file_id = uuid.uuid4().hex
        try:
            mime = sniff_mime_from_upload(uf)
            event_file_meta["mime"] = mime
            if not is_allowed(mime):
                result = UploadResult(status="error", message=f"Unsupported type: {mime}")
                results.append(result)
                _record_file_event(
                    name="files.upload",
                    user_id=user_id,
                    thread_id=thread_id,
                    metadata={
                        "status": result.status,
                        "message": result.message,
                        "file": event_file_meta,
                    },
                )
                continue

            # optional strictness: extension vs sniffed MIME family
            ext_mime = mimetypes.guess_type(uf.filename or "")[0]
            if ext_mime and ext_mime.split("/")[0] != mime.split("/")[0]:
                result = UploadResult(
                    status="error",
                    message=f"Extension/MIME mismatch: {ext_mime} vs {mime}",
                )
                results.append(result)
                _record_file_event(
                    name="files.upload",
                    user_id=user_id,
                    thread_id=thread_id,
                    metadata={
                        "status": result.status,
                        "message": result.message,
                        "file": event_file_meta,
                    },
                )
                continue

            # compute hash for dedup
            sha = sha256_streaming(uf.file)
            event_file_meta["sha256"] = sha

            # check for existing (same user, thread, sha, name)
            dup = catalog.get_by_sha_and_name(user_id, thread_id, sha, uf.filename or "")
            if dup:
                result = UploadResult(
                    status="skipped",
                    message="Duplicate detected (same name & SHA-256).",
                    file=dup,
                )
                results.append(result)
                _record_file_event(
                    name="files.upload",
                    user_id=user_id,
                    thread_id=thread_id,
                    metadata={
                        "status": result.status,
                        "message": result.message,
                        "file": dup.model_dump(exclude_none=True),
                        "duplicate": True,
                    },
                )
                continue

            # write with server-side size enforcement
            path = write_stream_atomic_with_limit(root, file_id, uf.file, max_bytes)

            meta = FileMeta(
                id=file_id,
                user_id=user_id,
                thread_id=thread_id,
                tenant_id=None,  # ← no longer used for partitioning; keep field for schema compatibility
                original_name=(uf.filename or "file"),
                mime=mime,
                size=path.stat().st_size,
                sha256=sha,
                path=str(path),  # Field excluded from API serialization in schema
                created_at=int(path.stat().st_mtime),
                indexed=False,
            )
            catalog.save_metadata(meta)

            msg = "Saved"
            if settings.AUTO_INGEST_UPLOADS:
                # plug ingestion here; on success:
                meta.indexed = True
                catalog.save_metadata(meta)
                msg = "Saved & indexed"

            result = UploadResult(status="stored", message=msg, file=meta)
            results.append(result)
            _record_file_event(
                name="files.upload",
                user_id=user_id,
                thread_id=thread_id,
                metadata={
                    "status": result.status,
                    "message": result.message,
                    "file": meta.model_dump(exclude_none=True),
                },
            )

        except ValueError as ve:
            result = UploadResult(status="error", message=str(ve))
            results.append(result)
            _record_file_event(
                name="files.upload",
                user_id=user_id,
                thread_id=thread_id,
                metadata={
                    "status": result.status,
                    "message": result.message,
                    "file": event_file_meta,
                    "error": str(ve),
                },
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Unexpected error while storing upload '%s'", uf.filename)
            result = UploadResult(status="error", message="Internal error")
            results.append(result)
            _record_file_event(
                name="files.upload",
                user_id=user_id,
                thread_id=thread_id,
                metadata={
                    "status": result.status,
                    "message": result.message,
                    "file": event_file_meta,
                    "error": "internal_error",
                },
            )

    return results


@router.get("", response_model=ListFilesResponse)
async def list_files(request: Request, thread_id: str | None = None):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = _get_user_id_from_claims(request.state.user)
    items = catalog.list_metadata(user_id=user_id, thread_id=thread_id)
    _record_file_event(
        name="files.list",
        user_id=user_id,
        thread_id=thread_id,
        metadata={
            "file_count": len(items),
            "files_preview": [
                item.model_dump(exclude_none=True) for item in items[:20]
            ],
        },
    )
    return ListFilesResponse(items=items)


@router.get("/{file_id}")
async def get_file(
    request: Request,
    file_id: str,
    thread_id: str | None = None,
    download: bool = False,
):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = _get_user_id_from_claims(request.state.user)
    operation = "files.download" if download else "files.get"

    meta = catalog.get_metadata_by_id(user_id=user_id, file_id=file_id, thread_id=thread_id)
    if not meta:
        _record_file_event(
            name=operation,
            user_id=user_id,
            thread_id=thread_id,
            metadata={
                "status": "not_found",
                "file": {"id": file_id},
            },
        )
        raise HTTPException(status_code=404, detail="Not found")

    path = Path(meta.path)
    if not path.exists():
        _record_file_event(
            name=operation,
            user_id=user_id,
            thread_id=thread_id,
            metadata={
                "status": "missing_on_disk",
                "file": meta.model_dump(exclude_none=True),
            },
        )
        raise HTTPException(status_code=410, detail="File missing")

    if download:
        _record_file_event(
            name=operation,
            user_id=user_id,
            thread_id=thread_id,
            metadata={
                "status": "downloaded",
                "file": meta.model_dump(exclude_none=True),
            },
        )
        return FileResponse(path, media_type=meta.mime, filename=meta.original_name)

    _record_file_event(
        name=operation,
        user_id=user_id,
        thread_id=thread_id,
        metadata={
            "status": "metadata_returned",
            "file": meta.model_dump(exclude_none=True),
        },
    )
    return meta


@router.delete("/{file_id}", status_code=204)
async def delete_file(request: Request, file_id: str, thread_id: str | None = None):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = _get_user_id_from_claims(request.state.user)

    meta = catalog.get_metadata_by_id(user_id=user_id, file_id=file_id, thread_id=thread_id)
    event_metadata: dict[str, Any] = {"file": {"id": file_id}}
    status = "not_found"
    error: str | None = None
    if meta:
        event_metadata["file"] = meta.model_dump(exclude_none=True)
        try:
            Path(meta.path).unlink(missing_ok=True)
            status = "deleted"
        except Exception as exc:  # pragma: no cover - defensive guard
            error = "delete_failed"
            status = "metadata_removed"
            logger.warning(
                "Failed to remove file '%s' from disk: %s", meta.path, exc
            )
        finally:
            catalog.delete_metadata(user_id=user_id, file_id=file_id, thread_id=thread_id)
    _record_file_event(
        name="files.delete",
        user_id=user_id,
        thread_id=thread_id,
        metadata={
            "status": status,
            **event_metadata,
            **({"error": error} if error else {}),
        },
    )
    return
