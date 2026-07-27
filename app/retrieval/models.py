"""Pydantic contracts for hybrid retrieval."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchFilters(BaseModel):
    """Optional metadata constraints applied to retrieved chunks."""

    model_config = ConfigDict(frozen=True)

    document_name: str | None = None
    document_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)


class SearchRequest(BaseModel):
    """Validated input for a hybrid retrieval operation."""

    model_config = ConfigDict(frozen=True)

    query: str
    limit: int = Field(default=5, ge=1)
    filters: SearchFilters | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject blank queries while retaining meaningful whitespace internally."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class RetrievedChunk(BaseModel):
    """Metadata and text associated with one indexed chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_number: int
    text: str


class SearchResult(RetrievedChunk):
    """A chunk returned by one retrieval strategy."""

    source: Literal["dense", "sparse"]
    similarity_score: float | None = None
    bm25_score: float | None = None


class HybridResult(RetrievedChunk):
    """A fused chunk with source scores and RRF evidence."""

    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    similarity_score: float | None = None
    bm25_score: float | None = None
