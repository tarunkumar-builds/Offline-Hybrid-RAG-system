"""Unit tests for the local cross-encoder reranking pipeline."""

import numpy as np

from app.reranker.config import RerankerConfig
from app.reranker.models import RerankCandidate
from app.reranker.pipeline import RerankingPipeline
from app.reranker.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Predictable cross-encoder stand-in for isolated pipeline tests."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def predict(self, pairs: list[tuple[str, str]], batch_size: int, show_progress_bar: bool) -> np.ndarray:
        self.calls.append((pairs, batch_size))
        return np.asarray(self.scores[: len(pairs)], dtype=np.float32)


def test_reranker_loads_model_once_and_batches_inference() -> None:
    loaded_models: list[FakeCrossEncoder] = []

    def loader(model_name: str, device: str) -> FakeCrossEncoder:
        assert model_name == "local-reranker"
        assert device == "cpu"
        model = FakeCrossEncoder([0.2, 0.8])
        loaded_models.append(model)
        return model

    reranker = CrossEncoderReranker(
        RerankerConfig(model_name="local-reranker", batch_size=2), model_loader=loader
    )

    reranker.score("query", [_candidate("one"), _candidate("two")])
    reranker.score("query", [_candidate("one")])

    assert len(loaded_models) == 1
    assert loaded_models[0].calls[0][1] == 2


def test_pipeline_sorts_scores_and_selects_top_n() -> None:
    pipeline = RerankingPipeline(
        RerankerConfig(top_k_input=3, top_n_output=2),
        reranker=FakeReranker([0.1, 0.9, 0.5]),
    )

    results = pipeline.rerank("query", [_candidate("one"), _candidate("two"), _candidate("three")])

    assert [result.chunk_id for result in results] == ["two", "three"]
    assert [result.rank for result in results] == [1, 2]
    assert round(results[0].rerank_score, 6) == 0.9


def test_pipeline_limits_cross_encoder_input_to_top_k() -> None:
    reranker = FakeReranker([0.3, 0.7])
    pipeline = RerankingPipeline(RerankerConfig(top_k_input=2, top_n_output=1), reranker=reranker)

    results = pipeline.rerank("query", [_candidate("one"), _candidate("two"), _candidate("three")])

    assert len(reranker.candidates) == 2
    assert [result.chunk_id for result in results] == ["two"]


def test_pipeline_returns_empty_results_without_inference() -> None:
    reranker = FakeReranker([])
    pipeline = RerankingPipeline(RerankerConfig(), reranker=reranker)

    assert pipeline.rerank("query", []) == []
    assert reranker.candidates == []


class FakeReranker:
    """Pipeline dependency that returns controlled relevance scores."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.candidates: list[RerankCandidate] = []

    def score(self, query: str, candidates: tuple[RerankCandidate, ...]) -> np.ndarray:
        self.candidates = list(candidates)
        return np.asarray(self._scores, dtype=np.float32)


def _candidate(chunk_id: str) -> RerankCandidate:
    return RerankCandidate(
        chunk_id=chunk_id,
        document_name="guide.pdf",
        page_number=1,
        chunk_number=0,
        text=f"content for {chunk_id}",
        retrieval_score=0.01,
    )
