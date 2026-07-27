"""Pydantic contracts for offline RAG evaluation."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.generation.models import Citation, GeneratedAnswer
from app.reranker.models import RerankedChunk
from app.retrieval.models import HybridResult


class EvaluationInput(BaseModel):
    """All artifacts needed to evaluate one completed RAG query."""

    model_config = ConfigDict(frozen=True)

    question: str
    retrieved_chunks: tuple[HybridResult, ...] = ()
    reranked_chunks: tuple[RerankedChunk, ...] = ()
    generated_answer: GeneratedAnswer | None = None
    citations: tuple[Citation, ...] = ()
    timings: dict[str, float] = Field(default_factory=dict)
    reference_answer: str | None = None
    expected_documents: tuple[str, ...] = ()

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject blank benchmark questions."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class RetrievalMetrics(BaseModel):
    """Quality and size measures for retrieved and reranked context."""

    retrieved_chunk_count: int = Field(ge=0)
    average_retrieval_score: float = 0.0
    average_rerank_score: float = 0.0
    duplicate_chunk_count: int = Field(ge=0)
    unique_document_count: int = Field(ge=0)
    average_chunk_length: float = 0.0
    context_size: int = Field(ge=0)


class GenerationMetrics(BaseModel):
    """Offline answer and optional reference-answer metrics."""

    answer_length: int = Field(ge=0)
    generation_time: float = Field(ge=0.0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    answer_available: bool
    empty_answer: bool
    unsupported_answer: bool
    exact_match: float | None = None
    rouge_l: float | None = None
    bleu: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


class CitationMetrics(BaseModel):
    """Coverage and duplication measures for answer citations."""

    citation_count: int = Field(ge=0)
    unique_source_documents: int = Field(ge=0)
    unique_pages: int = Field(ge=0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    missing_citations: bool
    duplicate_citations: int = Field(ge=0)


class PerformanceMetrics(BaseModel):
    """Stage timings expressed in seconds."""

    embedding_time: float = Field(default=0.0, ge=0.0)
    retrieval_latency: float = Field(default=0.0, ge=0.0)
    bm25_latency: float = Field(default=0.0, ge=0.0)
    faiss_latency: float = Field(default=0.0, ge=0.0)
    fusion_latency: float = Field(default=0.0, ge=0.0)
    reranking_latency: float = Field(default=0.0, ge=0.0)
    prompt_construction_time: float = Field(default=0.0, ge=0.0)
    llm_generation_time: float = Field(default=0.0, ge=0.0)
    total_pipeline_time: float = Field(default=0.0, ge=0.0)


class EvaluationResult(BaseModel):
    """Complete metric output for one evaluated query."""

    question: str
    answer: str | None
    retrieved_documents: tuple[str, ...]
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    citations: CitationMetrics
    performance: PerformanceMetrics


class DatasetRecord(BaseModel):
    """A benchmark row with optional reference answer and expected documents."""

    question: str
    reference_answer: str | None = None
    expected_documents: tuple[str, ...] = ()


class BenchmarkSummary(BaseModel):
    """Aggregated benchmark measurements."""

    total_queries: int = Field(ge=0)
    successful_queries: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    average_latency: float = Field(ge=0.0)
    average_citation_coverage: float = Field(ge=0.0, le=1.0)
    best_question: str | None = None
    worst_question: str | None = None


class BenchmarkResult(BaseModel):
    """All per-query results and their computed summary."""

    results: tuple[EvaluationResult, ...]
    summary: BenchmarkSummary
