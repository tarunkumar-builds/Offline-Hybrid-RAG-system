"""Pydantic contracts for cross-encoder reranking."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.retrieval.models import HybridResult


class RerankCandidate(BaseModel):
    """A retrieved chunk supplied to the cross-encoder."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_name: str
    page_number: int = Field(ge=1)
    chunk_number: int = Field(ge=0)
    text: str = Field(min_length=1)
    retrieval_score: float

    @classmethod
    def from_hybrid_result(cls, result: HybridResult) -> "RerankCandidate":
        """Adapt a Phase 2 fused result to the reranker input contract."""
        return cls(
            chunk_id=result.chunk_id,
            document_name=result.document_name,
            page_number=result.page_number,
            chunk_number=result.chunk_number,
            text=result.text,
            retrieval_score=result.rrf_score,
        )


class RerankRequest(BaseModel):
    """Validated query and candidate collection for reranking."""

    model_config = ConfigDict(frozen=True)

    query: str
    candidates: tuple[RerankCandidate, ...]

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject blank queries before expensive inference starts."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class RerankedChunk(RerankCandidate):
    """A candidate scored and ranked by the cross-encoder."""

    rerank_score: float
    rank: int = Field(ge=1)
