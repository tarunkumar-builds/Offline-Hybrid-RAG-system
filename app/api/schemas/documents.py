"""Document API request and response models."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Public metadata for an indexed document."""

    document_id: str
    document_name: str
    ingestion_time: datetime
    chunk_count: int = Field(ge=0)


class UploadResponse(BaseModel):
    """Result of a successful PDF ingestion request."""

    documents: list[DocumentResponse]
    indexed_chunks: int = Field(ge=0)
