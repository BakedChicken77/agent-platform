## src/service/files_router.py

from __future__ import annotations

import mimetypes
import uuid
from collections.abc import Mapping
from pathlib import Path

import filetype
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from core import settings
from core.langgraph import (
    emit_runtime_event,
    end_runtime_span,
    runtime_metadata,
    start_runtime_span,
    update_runtime_span,
)
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

LANGFUSE_TRACE_HEADER = "x-langfuse-trace-id"
LANGFUSE_SESSION_HEADER = "x-langfuse-session-id"
LANGFUSE_RUN_HEADER = "x-langfuse-run-id"


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


def _build_langfuse_runtime(
    request: Request,
    *,
    user_id: str,
    thread_id: str | None,
) -> dict[str, str]:
    """Derive Langfuse runtime context from headers/query parameters."""

    runtime: dict[str, str] = {"user_id": user_id}
    if thread_id:
        runtime["thread_id"] = thread_id

    query_params = request.query_params
    headers = request.headers

    trace_id = (
        query_params.get("trace_id")
        or headers.get(LANGFUSE_TRACE_HEADER)
        or ""
    ).strip()
    session_id = (
        query_params.get("session_id")
        or headers.get(LANGFUSE_SESSION_HEADER)
        or ""
    ).strip()
    run_id = (
        query_params.get("run_id")
        or headers.get(LANGFUSE_RUN_HEADER)
        or ""
    ).strip()

    if trace_id:
        runtime["trace_id"] = trace_id
    if session_id:
        runtime["session_id"] = session_id
    elif thread_id:
        runtime["session_id"] = thread_id
    if run_id:
        runtime["run_id"] = run_id

    return runtime


def _start_file_span(
    name: str,
    runtime: Mapping[str, str] | None,
    **metadata: str | int | float | None,
):
    meta = runtime_metadata(runtime, **{k: v for k, v in metadata.items() if v is not None})
    span, context = start_runtime_span(name, runtime, metadata=meta)
    return span, context, meta


def _emit_file_event(
    name: str,
    runtime: Mapping[str, str] | None,
    **metadata: str | int | float | None,
) -> None:
    meta = runtime_metadata(runtime, **{k: v for k, v in metadata.items() if v is not None})
    emit_runtime_event(name, runtime, metadata=meta)


@router.post("/upload", response_model=list[UploadResult])
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    thread_id: str | None = None,
):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = _get_user_id_from_claims(request.state.user)
    runtime = _build_langfuse_runtime(request, user_id=user_id, thread_id=thread_id)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Use (user_id, thread_id) for storage partitioning; do not use tenant_id.
    root = build_user_dir(user_id=user_id, thread_id=thread_id, tenant_id=None)
    results: list[UploadResult] = []
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    span, context, _ = _start_file_span(
        "service.files.upload",
        runtime,
        file_count=len(files),
        thread_id=thread_id,
    )
    error: Exception | None = None

    try:
        with context:
            for uf in files:
                file_id = uuid.uuid4().hex
                filename = uf.filename or ""
                file_size: int | None = None
                try:
                    mime = sniff_mime_from_upload(uf)
                    if not is_allowed(mime):
                        results.append(UploadResult(status="error", message=f"Unsupported type: {mime}"))
                        _emit_file_event(
                            "service.files.upload.file",
                            runtime,
                            status="error",
                            reason="unsupported_mime",
                            filename=filename,
                            mime=mime,
                        )
                        continue

                    # optional strictness: extension vs sniffed MIME family
                    ext_mime = mimetypes.guess_type(filename)[0]
                    if ext_mime and ext_mime.split("/")[0] != mime.split("/")[0]:
                        results.append(
                            UploadResult(
                                status="error",
                                message=f"Extension/MIME mismatch: {ext_mime} vs {mime}",
                            )
                        )
                        _emit_file_event(
                            "service.files.upload.file",
                            runtime,
                            status="error",
                            reason="mime_mismatch",
                            filename=filename,
                            mime=mime,
                            ext_mime=ext_mime,
                        )
                        continue

                    # compute hash for dedup
                    sha = sha256_streaming(uf.file)

                    # check for existing (same user, thread, sha, name)
                    dup = catalog.get_by_sha_and_name(user_id, thread_id, sha, filename)
                    if dup:
                        results.append(
                            UploadResult(
                                status="skipped",
                                message="Duplicate detected (same name & SHA-256).",
                                file=dup,
                            )
                        )
                        _emit_file_event(
                            "service.files.upload.file",
                            runtime,
                            status="skipped",
                            reason="duplicate",
                            filename=filename,
                            file_id=dup.id,
                            sha256=sha,
                            file_size=dup.size,
                        )
                        continue

                    # write with server-side size enforcement
                    path = write_stream_atomic_with_limit(root, file_id, uf.file, max_bytes)
                    file_size = path.stat().st_size

                    meta = FileMeta(
                        id=file_id,
                        user_id=user_id,
                        thread_id=thread_id,
                        tenant_id=None,  # ← no longer used for partitioning; keep field for schema compatibility
                        original_name=(filename or "file"),
                        mime=mime,
                        size=file_size,
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

                    results.append(UploadResult(status="stored", message=msg, file=meta))
                    _emit_file_event(
                        "service.files.upload.file",
                        runtime,
                        status="stored",
                        filename=filename,
                        file_id=file_id,
                        file_size=file_size,
                        mime=mime,
                        sha256=sha,
                        indexed=meta.indexed,
                    )

                except ValueError as ve:
                    results.append(UploadResult(status="error", message=str(ve)))
                    _emit_file_event(
                        "service.files.upload.file",
                        runtime,
                        status="error",
                        reason="validation",
                        filename=filename,
                        file_id=file_id,
                        file_size=file_size,
                        error=str(ve),
                    )
                except Exception as exc:
                    results.append(UploadResult(status="error", message="Internal error"))
                    _emit_file_event(
                        "service.files.upload.file",
                        runtime,
                        status="error",
                        reason="exception",
                        filename=filename,
                        file_id=file_id,
                        file_size=file_size,
                        error=str(exc),
                    )

        stored = sum(1 for item in results if item.status == "stored")
        skipped = sum(1 for item in results if item.status == "skipped")
        errors = sum(1 for item in results if item.status == "error")
        update_runtime_span(span, stored=stored, skipped=skipped, errors=errors)

    except Exception as exc:  # pragma: no cover - defensive guard
        error = exc
        raise
    finally:
        end_runtime_span(span, error=error)

    return results


@router.get("", response_model=ListFilesResponse)
async def list_files(request: Request, thread_id: str | None = None):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = _get_user_id_from_claims(request.state.user)
    runtime = _build_langfuse_runtime(request, user_id=user_id, thread_id=thread_id)
    items = catalog.list_metadata(user_id=user_id, thread_id=thread_id)

    span, context, _ = _start_file_span(
        "service.files.list",
        runtime,
        thread_id=thread_id,
    )
    error: Exception | None = None
    try:
        with context:
            update_runtime_span(span, count=len(items))
            _emit_file_event(
                "service.files.list",
                runtime,
                status="success",
                count=len(items),
                thread_id=thread_id,
            )
    except Exception as exc:  # pragma: no cover - defensive guard
        error = exc
        raise
    finally:
        end_runtime_span(span, error=error)

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
    runtime = _build_langfuse_runtime(request, user_id=user_id, thread_id=thread_id)

    meta = catalog.get_metadata_by_id(user_id=user_id, file_id=file_id, thread_id=thread_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Not found")

    path = Path(meta.path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="File missing")

    span, context, _ = _start_file_span(
        "service.files.download" if download else "service.files.get",
        runtime,
        file_id=file_id,
        filename=meta.original_name,
        file_size=meta.size,
    )
    error: Exception | None = None
    response = meta
    try:
        with context:
            if download:
                response = FileResponse(path, media_type=meta.mime, filename=meta.original_name)
                event_name = "service.files.download"
            else:
                event_name = "service.files.get"
            _emit_file_event(
                event_name,
                runtime,
                status="success",
                file_id=file_id,
                filename=meta.original_name,
                file_size=meta.size,
                download=download,
            )
            return response
    except Exception as exc:  # pragma: no cover - defensive guard
        error = exc
        raise
    finally:
        end_runtime_span(span, error=error)


@router.delete("/{file_id}", status_code=204)
async def delete_file(request: Request, file_id: str, thread_id: str | None = None):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = _get_user_id_from_claims(request.state.user)
    runtime = _build_langfuse_runtime(request, user_id=user_id, thread_id=thread_id)

    meta = catalog.get_metadata_by_id(user_id=user_id, file_id=file_id, thread_id=thread_id)
    span, context, _ = _start_file_span(
        "service.files.delete",
        runtime,
        file_id=file_id,
        filename=(meta.original_name if meta else None),
    )
    error: Exception | None = None
    try:
        with context:
            if meta:
                try:
                    Path(meta.path).unlink(missing_ok=True)
                finally:
                    catalog.delete_metadata(user_id=user_id, file_id=file_id, thread_id=thread_id)
                _emit_file_event(
                    "service.files.delete",
                    runtime,
                    status="success",
                    file_id=file_id,
                    filename=meta.original_name,
                )
            else:
                _emit_file_event(
                    "service.files.delete",
                    runtime,
                    status="missing",
                    file_id=file_id,
                )
    except Exception as exc:  # pragma: no cover - defensive guard
        error = exc
        raise
    finally:
        end_runtime_span(span, error=error)
    return
