"""Unit tests for the Phase 2 hybrid retrieval components."""

from pathlib import Path
from typing import Literal

import numpy as np

from app.database.metadata import MetadataStore
from app.ingestion.faiss_index import FaissIndexManager
from app.models import Chunk, DocumentRecord
from app.retrieval.dense_search import DenseSearcher
from app.retrieval.filters import apply_filters
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import SearchFilters, SearchResult
from app.retrieval.sparse_search import SparseSearcher


def _metadata_store(tmp_path: Path) -> MetadataStore:
    store = MetadataStore(tmp_path / "metadata.db")
    store.initialize()
    store.add_document(
        DocumentRecord(
            document_id="doc-1",
            document_name="guide.pdf",
            file_path=tmp_path / "guide.pdf",
            file_hash="hash-1",
            ingestion_time="2026-01-01T00:00:00Z",
        ),
        [
            Chunk(chunk_id="chunk-1", document_id="doc-1", page_number=1, chunk_number=0, text="FAISS vectors"),
            Chunk(chunk_id="chunk-2", document_id="doc-1", page_number=2, chunk_number=1, text="BM25 keywords"),
        ],
        [10, 11],
        embedding_dimension=3,
    )
    return store


def test_dense_search_returns_matching_chunk(tmp_path: Path) -> None:
    store = _metadata_store(tmp_path)
    index_path = tmp_path / "vectors.faiss"
    index = FaissIndexManager(index_path, dimension=3)
    index.add(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), np.array([10, 11]))
    index.save()

    results = DenseSearcher(index_path, tmp_path / "metadata.db", store).search(
        np.array([[1, 0, 0]], dtype=np.float32), limit=1
    )

    assert results[0].chunk_id == "chunk-1"
    assert results[0].similarity_score == 1.0


def test_bm25_search_returns_keyword_match(tmp_path: Path) -> None:
    store = _metadata_store(tmp_path)

    results = SparseSearcher(store, tmp_path / "metadata.db").search("keywords", limit=1)

    assert results[0].chunk_id == "chunk-2"
    assert results[0].bm25_score is not None


def test_rrf_removes_duplicates_and_combines_scores() -> None:
    dense = [_result("chunk-1", "dense", similarity_score=0.9), _result("chunk-2", "dense", similarity_score=0.8)]
    sparse = [_result("chunk-1", "sparse", bm25_score=3.0)]

    fused = reciprocal_rank_fusion(dense, sparse)

    assert [result.chunk_id for result in fused] == ["chunk-1", "chunk-2"]
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 1
    assert fused[0].rrf_score == 2 / 61


def test_filters_restrict_results_by_document_and_page() -> None:
    results = [_result("chunk-1", "dense", page_number=1), _result("chunk-2", "dense", page_number=2)]

    filtered = apply_filters(results, SearchFilters(document_name="guide.pdf", page_number=2))

    assert [result.chunk_id for result in filtered] == ["chunk-2"]


def _result(
    chunk_id: str,
    source: Literal["dense", "sparse"],
    page_number: int = 1,
    similarity_score: float | None = None,
    bm25_score: float | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="guide.pdf",
        page_number=page_number,
        chunk_number=0,
        text="retrieval content",
        source=source,
        similarity_score=similarity_score,
        bm25_score=bm25_score,
    )
