"""Reusable query-to-reranked-context pipeline."""

from collections.abc import Sequence
from time import perf_counter

from loguru import logger

from app.config.settings import Settings
from app.reranker.cache import RerankCache
from app.reranker.config import RerankerConfig
from app.reranker.models import RerankCandidate, RerankedChunk
from app.reranker.reranker import CrossEncoderReranker
from app.retrieval.models import HybridResult


class RerankingPipeline:
    """Limit hybrid candidates, score them, then return the highest quality chunks."""

    def __init__(
        self,
        config: RerankerConfig,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._config = config
        self._reranker = reranker or CrossEncoderReranker.get_instance(config)
        self._cache: RerankCache[str, tuple[RerankedChunk, ...]] = RerankCache(config.cache_size)

    @classmethod
    def from_settings(cls, settings: Settings) -> "RerankingPipeline":
        """Build a pipeline using shared application configuration."""
        return cls(RerankerConfig.from_settings(settings))

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate | HybridResult],
        top_n: int | None = None,
    ) -> list[RerankedChunk]:
        """Cross-encode top retrieved chunks and return them in relevance order."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        output_limit = top_n or self._config.top_n_output
        if output_limit < 1:
            raise ValueError("top_n must be at least one")
        normalized_candidates = tuple(self._to_candidate(candidate) for candidate in candidates)
        selected_candidates = normalized_candidates[: self._config.top_k_input]
        if not selected_candidates:
            logger.info("Reranking skipped because retrieval returned no candidates")
            return []
        cache_key = self._cache_key(normalized_query, selected_candidates, output_limit)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        logger.info("Reranking query received: {}", normalized_query)
        logger.info("Reranking {} candidates with batch size {}", len(selected_candidates), self._config.batch_size)
        start = perf_counter()
        scores = self._reranker.score(normalized_query, selected_candidates)
        ordered = sorted(
            zip(selected_candidates, scores, strict=True), key=lambda item: float(item[1]), reverse=True
        )[:output_limit]
        results = [
            RerankedChunk(**candidate.model_dump(), rerank_score=float(score), rank=rank)
            for rank, (candidate, score) in enumerate(ordered, start=1)
        ]
        logger.info("Cross-encoder reranking completed in {:.3f}s", perf_counter() - start)
        self._cache.put(cache_key, tuple(results))
        return results

    @staticmethod
    def _to_candidate(candidate: RerankCandidate | HybridResult) -> RerankCandidate:
        if isinstance(candidate, RerankCandidate):
            return candidate
        return RerankCandidate.from_hybrid_result(candidate)

    @staticmethod
    def _cache_key(query: str, candidates: tuple[RerankCandidate, ...], top_n: int) -> str:
        return f"{query}|{top_n}|" + "|".join(candidate.model_dump_json() for candidate in candidates)
