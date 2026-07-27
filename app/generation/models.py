"""Pydantic response contracts for grounded local answer generation."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.reranker.models import RerankedChunk


class Citation(BaseModel):
    """A cited source chunk used to support a generated answer."""

    model_config = ConfigDict(frozen=True)

    document_name: str
    page_number: int = Field(ge=1)
    chunk_number: int = Field(ge=0)
    snippet: str


class GenerationRequest(BaseModel):
    """Validated user question and reranked context for generation."""

    model_config = ConfigDict(frozen=True)

    question: str
    context_chunks: tuple[RerankedChunk, ...]

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject blank questions before creating a prompt."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class GeneratedAnswer(BaseModel):
    """Complete answer payload returned by the offline generation engine."""

    answer: str = Field(min_length=1)
    citations: tuple[Citation, ...]
    source_documents: tuple[str, ...]
    retrieved_chunks: tuple[RerankedChunk, ...]
    model_name: str
    generation_time: float = Field(ge=0.0)
    prompt_tokens: int = Field(ge=0)
    response_tokens: int = Field(ge=0)
