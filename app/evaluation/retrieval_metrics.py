"""Offline retrieval and context metric calculations."""

from collections.abc import Sequence

from app.evaluation.models import RetrievalMetrics
from app.reranker.models import RerankedChunk
from app.retrieval.models import HybridResult


def calculate_retrieval_metrics(
    retrieved_chunks: Sequence[HybridResult], reranked_chunks: Sequence[RerankedChunk]
) -> RetrievalMetrics:
    """Summarize candidate volume, score distributions, duplicates, and context size."""
    all_chunks = [*retrieved_chunks, *reranked_chunks]
    retrieval_scores = [chunk.rrf_score for chunk in retrieved_chunks]
    rerank_scores = [chunk.rerank_score for chunk in reranked_chunks]
    chunk_ids = [chunk.chunk_id for chunk in all_chunks]
    texts = [chunk.text for chunk in all_chunks]
    return RetrievalMetrics(
        retrieved_chunk_count=len(retrieved_chunks),
        average_retrieval_score=_average(retrieval_scores),
        average_rerank_score=_average(rerank_scores),
        duplicate_chunk_count=len(chunk_ids) - len(set(chunk_ids)),
        unique_document_count=len({chunk.document_name for chunk in all_chunks}),
        average_chunk_length=_average([len(text) for text in texts]),
        context_size=sum(len(text) for text in texts),
    )


def _average(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
