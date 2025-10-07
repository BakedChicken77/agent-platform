from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import BinaryIO

from core.settings import settings

SAFE_NAME_CHARS = "-_.() "
MAX_CHUNK = 1024 * 1024  # 1 MiB


def ensure_upload_root() -> Path:
    root = Path(settings.UPLOAD_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize(name: str) -> str:
    # keep ascii letters/digits & SAFE_NAME_CHARS; replace others with underscore
    return "".join(c if (c.isalnum() or c in SAFE_NAME_CHARS) else "_" for c in name).strip() or "file"


def sha256_streaming(fp: BinaryIO) -> str:
    fp.seek(0)
    h = hashlib.sha256()
    for chunk in iter(lambda: fp.read(MAX_CHUNK), b""):
        h.update(chunk)
    fp.seek(0)
    return h.hexdigest()


def is_allowed(mime: str) -> bool:
    return mime in settings.ALLOWED_MIME_TYPES


def enforce_size_limit(size_bytes: int) -> None:
    if size_bytes > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds max size of {settings.MAX_UPLOAD_MB} MB")


def build_user_dir(user_id: str, thread_id: str | None = None, tenant_id: str | None = None) -> Path:
    base = ensure_upload_root()
    parts = [p for p in [tenant_id, user_id, thread_id] if p]
    target = base.joinpath(*parts) if parts else base
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_bytes_atomic(target_dir: Path, file_id: str, data: bytes) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    final = target_dir / file_id
    tmp = target_dir / f".{file_id}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(final)
    return final


def write_stream_atomic(target_dir: Path, file_id: str, src: BinaryIO) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    final = target_dir / file_id
    tmp = target_dir / f".{file_id}.tmp"
    with open(tmp, "wb") as dst:
        for chunk in iter(lambda: src.read(MAX_CHUNK), b""):
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    tmp.replace(final)
    return final

def write_stream_atomic_with_limit(target_dir: Path, file_id: str, src: BinaryIO, max_bytes: int) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    final = target_dir / file_id
    tmp = target_dir / f".{file_id}.tmp"
    total = 0
    with open(tmp, "wb") as dst:
        for chunk in iter(lambda: src.read(MAX_CHUNK), b""):
            total += len(chunk)
            if total > max_bytes:
                dst.close()
                tmp.unlink(missing_ok=True)
                raise ValueError(f"File exceeds max size of {settings.MAX_UPLOAD_MB} MB")
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    tmp.replace(final)
    return final
