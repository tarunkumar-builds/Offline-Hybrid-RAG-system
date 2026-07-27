"""End-to-end offline retrieval, reranking, and answer generation pipeline."""

from app.config.settings import Settings
from app.generation.answer_generator import AnswerGenerator
from app.generation.config import GenerationConfig
from app.generation.models import GeneratedAnswer
from app.reranker.pipeline import RerankingPipeline
from app.retrieval.models import SearchRequest
from app.retrieval.retriever import HybridRetriever


class GenerationPipeline:
    """Produce grounded local answers from a user question in one reusable call."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranking_pipeline: RerankingPipeline,
        answer_generator: AnswerGenerator,
        reranker_input_limit: int,
    ) -> None:
        self._retriever = retriever
        self._reranking_pipeline = reranking_pipeline
        self._answer_generator = answer_generator
        self._reranker_input_limit = reranker_input_limit

    @classmethod
    def from_settings(cls, settings: Settings) -> "GenerationPipeline":
        """Assemble the complete Phase 1-4 pipeline from shared settings."""
        config = GenerationConfig.from_settings(settings)
        return cls(
            retriever=HybridRetriever(settings),
            reranking_pipeline=RerankingPipeline.from_settings(settings),
            answer_generator=AnswerGenerator(config),
            reranker_input_limit=settings.reranker_top_k_input,
        )

    def answer(self, question: str) -> GeneratedAnswer:
        """Retrieve, rerank, and generate a citation-backed answer."""
        candidates = self._retriever.search(SearchRequest(query=question, limit=self._reranker_input_limit))
        context_chunks = self._reranking_pipeline.rerank(question, candidates)
        return self._answer_generator.generate(question, context_chunks)

    def refresh(self) -> None:
        """Discard retrieval state after the document corpus changes."""
        self._retriever.refresh()

    def close(self) -> None:
        """Release externally managed resources used by this pipeline."""
        self._answer_generator.close()
