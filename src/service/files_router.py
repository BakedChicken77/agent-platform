from __future__ import annotations

import mimetypes
import filetype
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from core import settings
from service.storage import (
    build_user_dir,
    is_allowed,
    sha256_streaming,
    write_stream_atomic_with_limit

)
from schema.files import FileMeta, ListFilesResponse, UploadResult
from service import catalog_postgres as catalog 

router = APIRouter(tags=["files"])


SNIFF_BYTES = 8192  # read a small header; safe for large files

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

    if kind and kind.mime:
        return kind.mime

    # fallback to filename extension
    ext_mime = mimetypes.guess_type(uf.filename or "")[0]
    if ext_mime:
        return ext_mime

    # final fallback to client-provided content_type (untrusted)
    return uf.content_type or "application/octet-stream"



def _get_ids_from_claims(claims: dict) -> tuple[str | None, str]:
    # Standard Graph token has tenant in "tid"
    tenant_id = claims.get("tid") or claims.get("tenant_id")
    user = claims.get("oid") or claims.get("sub") or claims.get("preferred_username") or "unknown"
    return tenant_id, str(user)


@router.post("/upload", response_model=list[UploadResult])
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    thread_id: str | None = None,
):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    tenant_id, user_id = _get_ids_from_claims(request.state.user)
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    root = build_user_dir(user_id=user_id, thread_id=thread_id, tenant_id=tenant_id)
    results: list[UploadResult] = []
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    for uf in files:
        file_id = uuid.uuid4().hex
        try:
            mime = sniff_mime_from_upload(uf)
            if not is_allowed(mime):
                results.append(UploadResult(status="error", message=f"Unsupported type: {mime}"))
                continue

            # optional strictness: extension vs sniffed MIME family
            ext_mime = mimetypes.guess_type(uf.filename or "")[0]
            if ext_mime and ext_mime.split("/")[0] != mime.split("/")[0]:
                results.append(UploadResult(status="error", message=f"Extension/MIME mismatch: {ext_mime} vs {mime}"))
                continue

            # compute hash for dedup
            sha = sha256_streaming(uf.file)

            # check for existing (same user, thread, sha, name)
            dup = catalog.get_by_sha_and_name(user_id, thread_id, sha, uf.filename or "")
            if dup:
                results.append(UploadResult(status="skipped", message="Duplicate detected (same name & SHA-256).", file=dup))
                continue

            # write with server-side size enforcement
            path = write_stream_atomic_with_limit(root, file_id, uf.file, max_bytes)

            meta = FileMeta(
                id=file_id,
                user_id=user_id,
                thread_id=thread_id,
                tenant_id=tenant_id,
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

            results.append(UploadResult(status="stored", message=msg, file=meta))

        except ValueError as ve:
            results.append(UploadResult(status="error", message=str(ve)))
        except Exception:
            results.append(UploadResult(status="error", message="Internal error"))

    return results


@router.get("", response_model=ListFilesResponse)
async def list_files(request: Request, thread_id: str | None = None):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    tenant_id, user_id = _get_ids_from_claims(request.state.user)

    items = catalog.list_metadata(user_id=user_id, thread_id=thread_id)
    return ListFilesResponse(items=items)


@router.get("/{file_id}")
async def get_file(request: Request, file_id: str, thread_id: str | None = None, download: bool = False):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    tenant_id, user_id = _get_ids_from_claims(request.state.user)

    meta = catalog.get_metadata_by_id(user_id=user_id, file_id=file_id, thread_id=thread_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Not found")

    path = Path(meta.path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="File missing")

    if download:
        return FileResponse(path, media_type=meta.mime, filename=meta.original_name)

    return meta


@router.delete("/{file_id}", status_code=204)
async def delete_file(request: Request, file_id: str, thread_id: str | None = None):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    tenant_id, user_id = _get_ids_from_claims(request.state.user)

    meta = catalog.get_metadata_by_id(user_id=user_id, file_id=file_id, thread_id=thread_id)
    if meta:
        try:
            Path(meta.path).unlink(missing_ok=True)
        finally:
            catalog.delete_metadata(user_id=user_id, file_id=file_id, thread_id=thread_id)
    return
