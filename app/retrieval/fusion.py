"""Reciprocal Rank Fusion for combining dense and sparse result lists."""

from collections.abc import Sequence

from app.retrieval.models import HybridResult, SearchResult


def reciprocal_rank_fusion(
    dense_results: Sequence[SearchResult],
    sparse_results: Sequence[SearchResult],
    k: int = 60,
) -> list[HybridResult]:
    """Fuse ranked result lists, de-duplicating chunks by their stable chunk ID."""
    if k < 1:
        raise ValueError("k must be at least one")
    aggregated: dict[str, dict[str, object]] = {}
    for source, results in (("dense", dense_results), ("sparse", sparse_results)):
        for rank, result in enumerate(results, start=1):
            record = aggregated.setdefault(
                result.chunk_id,
                {"result": result, "rrf_score": 0.0, "dense_rank": None, "sparse_rank": None,
                 "similarity_score": None, "bm25_score": None},
            )
            record["rrf_score"] = float(record["rrf_score"]) + 1 / (k + rank)
            record[f"{source}_rank"] = rank
            if source == "dense":
                record["similarity_score"] = result.similarity_score
            else:
                record["bm25_score"] = result.bm25_score
    fused = [
        HybridResult(
            **record["result"].model_dump(exclude={"source", "similarity_score", "bm25_score"}),
            rrf_score=float(record["rrf_score"]),
            dense_rank=record["dense_rank"],
            sparse_rank=record["sparse_rank"],
            similarity_score=record["similarity_score"],
            bm25_score=record["bm25_score"],
        )
        for record in aggregated.values()
    ]
    return sorted(fused, key=lambda result: result.rrf_score, reverse=True)
