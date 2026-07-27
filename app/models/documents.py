"""Pydantic models for documents and chunks."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class PageText(BaseModel):
    """Extracted text from one non-empty PDF page."""

    document_name: str
    document_path: Path
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)


class Chunk(BaseModel):
    """A searchable text unit produced from one PDF page."""

    chunk_id: str
    document_id: str
    page_number: int = Field(ge=1)
    chunk_number: int = Field(ge=0)
    text: str = Field(min_length=1)


class DocumentRecord(BaseModel):
    """Persisted document-level information."""

    document_id: str
    document_name: str
    file_path: Path
    file_hash: str
    ingestion_time: datetime


class StoredChunk(BaseModel):
    """A chunk joined with its document metadata and FAISS vector identifier."""

    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_number: int
    text: str
    vector_id: int
