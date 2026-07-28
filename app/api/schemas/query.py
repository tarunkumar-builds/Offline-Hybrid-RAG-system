"""Question-answering API contracts."""

from pydantic import BaseModel, Field, field_validator

from app.generation.models import Citation
retrieved_chunks: tuple[RetrievedChunk, ...]


class QueryRequest(BaseModel):
    """Request payload for the RAG query endpoint."""

    question: str = Field(
        ...,
        description="User question.",
        examples=["What is Retrieval-Augmented Generation?"],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of retrieved chunks.",
    )

    prompt_template: str | None = Field(
        default=None,
        description="Prompt template to use.",
    )

    model_name: str | None = Field(
        default=None,
        description="Override the default Ollama model.",
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM temperature.",
    )

    evaluation_enabled: bool = Field(
        default=False,
        description="Run the evaluation engine.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be blank.")

        return value


class QueryResponse(BaseModel):
    """Response returned by the RAG query endpoint."""

    answer: str = Field(
        description="Generated answer."
    )

    citations: tuple[Citation, ...] = Field(
        default_factory=tuple,
        description="Supporting citations."
    )

    retrieved_chunks: tuple[RetrievedChunk, ...] = Field(
        default_factory=tuple,
        description="Chunks used to generate the answer."
    )

    model_name: str = Field(
        description="LLM used for generation."
    )

    generation_time: float = Field(
        description="LLM generation time in seconds."
    )

    processing_time: float = Field(
        description="Total API processing time in seconds."
    )

    evaluation: dict[str, object] | None = Field(
        default=None,
        description="Optional evaluation metrics."
    )