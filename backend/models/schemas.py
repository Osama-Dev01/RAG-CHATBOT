from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# ── Upload ────

class UploadResponse(BaseModel):
    success: bool
    filename: str
    message: str
    chunks_created: int


class FileInfo(BaseModel):
    filename: str
    uploaded_at: str
    size_bytes: int


class FilesListResponse(BaseModel):
    files: List[FileInfo]
    total: int


# ── Chat ──────

class ChatRequest(BaseModel):
    question: str
    session_id: str


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_index: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[str]


class ChatMessage(BaseModel):
    role: str        # "human" or "assistant"
    content: str
    timestamp: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]


# ── Health ──

class HealthResponse(BaseModel):
    status: str
    message: str
