"""Question-answering API contracts."""

from pydantic import BaseModel, Field, field_validator

from app.evaluation.models import EvaluationResult
from app.generation.models import Citation
from app.reranker.models import RerankedChunk


class QueryRequest(BaseModel):
    """Validated controls for one grounded answer request."""

    question: str = Field(min_length=1, max_length=4_000, description="Question answered using indexed documents.")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of final reranked source chunks.")
    prompt_template: str | None = Field(default=None, max_length=64, description="Optional packaged prompt template.")
    model_name: str | None = Field(default=None, max_length=128, description="Optional locally installed Ollama model.")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    evaluation_enabled: bool = False

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()

    @field_validator("prompt_template", "model_name")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template(cls, value: str | None) -> str | None:
        if value is not None and not all(character.isalnum() or character in {"-", "_"} for character in value):
            raise ValueError("prompt_template may contain only letters, numbers, hyphens, and underscores")
        return value


class QueryResponse(BaseModel):
    """Grounded answer and request-level observability details."""

    answer: str
    citations: tuple[Citation, ...]
    retrieved_chunks: tuple[RerankedChunk, ...]
    model_name: str
    generation_time: float = Field(ge=0.0)
    processing_time: float = Field(ge=0.0)
    evaluation: EvaluationResult | None = None
