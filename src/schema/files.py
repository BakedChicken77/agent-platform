from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FileMeta(BaseModel):
    id: str
    user_id: str
    thread_id: str | None = None
    tenant_id: str | None = None
    original_name: str
    mime: str
    size: int
    sha256: str
    path: str = Field(repr=False, exclude=True)  # hide from API responses
    created_at: int
    indexed: bool = False


UploadStatus = Literal["stored", "skipped", "error"]


class UploadResult(BaseModel):
    status: UploadStatus
    message: str
    file: FileMeta | None = None


class ListFilesResponse(BaseModel):
    items: list[FileMeta]
