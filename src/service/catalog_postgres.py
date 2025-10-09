from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from core import settings
from schema.files import FileMeta


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Reuse a single SQLAlchemy engine for the file catalog.
    Uses PGVECTOR_URL already present in your settings.
    """
    if not settings.PGVECTOR_URL:
        raise RuntimeError("PGVECTOR_URL is not configured")
    return create_engine(settings.PGVECTOR_URL, pool_pre_ping=True)


def init() -> None:
    """
    Create the file catalog table + indexes if not present.
    BIGINT created_at (epoch seconds) keeps parity with your Pydantic model.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS user_files (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        thread_id TEXT NULL,
        tenant_id TEXT NULL,
        original_name TEXT NOT NULL,
        mime TEXT NOT NULL,
        size BIGINT NOT NULL,
        sha256 TEXT NOT NULL,
        path TEXT NOT NULL,
        created_at BIGINT NOT NULL,
        indexed BOOLEAN NOT NULL DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_user_files_user_thread
        ON user_files (user_id, thread_id);

    CREATE INDEX IF NOT EXISTS idx_user_files_user_created_at
        ON user_files (user_id, created_at DESC);

    CREATE INDEX IF NOT EXISTS idx_user_files_user_sha_name
        ON user_files (user_id, sha256, original_name);

    CREATE UNIQUE INDEX IF NOT EXISTS uq_user_thread_sha_name
        ON user_files (user_id, COALESCE(thread_id, ''), sha256, original_name);
    """
    eng = get_engine()
    with eng.begin() as conn:
        conn.exec_driver_sql(ddl)


def _row_to_meta(row: dict[str, Any]) -> FileMeta:
    return FileMeta(
        id=row["id"],
        user_id=row["user_id"],
        thread_id=row["thread_id"],
        tenant_id=row["tenant_id"],
        original_name=row["original_name"],
        mime=row["mime"],
        size=int(row["size"]),
        sha256=row["sha256"],
        path=row["path"],            # NOTE: Field is excluded from API responses via schema
        created_at=int(row["created_at"]),
        indexed=bool(row["indexed"]),
    )


def save_metadata(meta: FileMeta) -> None:
    q = text("""
        INSERT INTO user_files (
            id, user_id, thread_id, tenant_id, original_name, mime, size, sha256, path, created_at, indexed
        ) VALUES (
            :id, :user_id, :thread_id, :tenant_id, :original_name, :mime, :size, :sha256, :path, :created_at, :indexed
        )
        ON CONFLICT (id) DO UPDATE SET
            thread_id = EXCLUDED.thread_id,
            tenant_id = EXCLUDED.tenant_id,
            original_name = EXCLUDED.original_name,
            mime = EXCLUDED.mime,
            size = EXCLUDED.size,
            sha256 = EXCLUDED.sha256,
            path = EXCLUDED.path,
            created_at = EXCLUDED.created_at,
            indexed = EXCLUDED.indexed
    """)
    params = {
        "id": meta.id,
        "user_id": meta.user_id,
        "thread_id": meta.thread_id,
        "tenant_id": meta.tenant_id,
        "original_name": meta.original_name,
        "mime": meta.mime,
        "size": int(meta.size),
        "sha256": meta.sha256,
        "path": meta.path,                    # ← include explicitly
        "created_at": int(meta.created_at),
        "indexed": bool(meta.indexed),
    }
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(q, params)


def get_metadata_by_id(user_id: str, file_id: str, thread_id: str | None = None) -> FileMeta | None:
    eng = get_engine()
    with eng.begin() as conn:
        if thread_id is None:
            q = text("""
                SELECT * FROM user_files
                WHERE user_id = :user_id AND id = :id
                LIMIT 1
            """)
            row = conn.execute(q, {"user_id": user_id, "id": file_id}).mappings().first()
        else:
            q = text("""
                SELECT * FROM user_files
                WHERE user_id = :user_id AND id = :id AND thread_id = :thread_id
                LIMIT 1
            """)
            row = conn.execute(q, {"user_id": user_id, "id": file_id, "thread_id": thread_id}).mappings().first()
    return _row_to_meta(row) if row else None

def list_metadata(user_id: str, thread_id: str | None = None) -> list[FileMeta]:
    eng = get_engine()
    with eng.begin() as conn:
        if thread_id is None:
            q = text("""
                SELECT * FROM user_files
                WHERE user_id = :user_id
                ORDER BY created_at DESC
            """)
            rows = conn.execute(q, {"user_id": user_id}).mappings().all()
        else:
            q = text("""
                SELECT * FROM user_files
                WHERE user_id = :user_id AND thread_id = :thread_id
                ORDER BY created_at DESC
            """)
            rows = conn.execute(q, {"user_id": user_id, "thread_id": thread_id}).mappings().all()
    return [_row_to_meta(r) for r in rows]


def get_by_sha_and_name(user_id: str, thread_id: str | None, sha256: str, original_name: str) -> FileMeta | None:
    eng = get_engine()
    with eng.begin() as conn:
        if thread_id is None:
            q = text("""
                SELECT * FROM user_files
                WHERE user_id = :user_id AND sha256 = :sha256 AND original_name = :original_name
                LIMIT 1
            """)
            row = conn.execute(q, {
                "user_id": user_id, "sha256": sha256, "original_name": original_name
            }).mappings().first()
        else:
            q = text("""
                SELECT * FROM user_files
                WHERE user_id = :user_id AND thread_id = :thread_id
                  AND sha256 = :sha256 AND original_name = :original_name
                LIMIT 1
            """)
            row = conn.execute(q, {
                "user_id": user_id, "thread_id": thread_id,
                "sha256": sha256, "original_name": original_name,
            }).mappings().first()
    return _row_to_meta(row) if row else None


def delete_metadata(user_id: str, file_id: str, thread_id: str | None = None) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        if thread_id is None:
            q = text("""
                DELETE FROM user_files
                WHERE user_id = :user_id AND id = :id
            """)
            conn.execute(q, {"user_id": user_id, "id": file_id})
        else:
            q = text("""
                DELETE FROM user_files
                WHERE user_id = :user_id AND id = :id AND thread_id = :thread_id
            """)
            conn.execute(q, {"user_id": user_id, "id": file_id, "thread_id": thread_id})
