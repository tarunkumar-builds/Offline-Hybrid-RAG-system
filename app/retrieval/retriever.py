"""Orchestrator for offline dense, sparse, and fused retrieval."""

from time import perf_counter

from loguru import logger

from app.config.settings import Settings
from app.database.metadata import MetadataStore
from app.retrieval.cache import LRUCache
from app.retrieval.dense_search import DenseSearcher
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import HybridResult, SearchRequest
from app.retrieval.query_embedder import QueryEmbedder
from app.retrieval.sparse_search import SparseSearcher


class HybridRetriever:
    """Coordinate cached hybrid retrieval against locally persisted artifacts."""

    def __init__(
        self,
        settings: Settings,
        query_embedder: QueryEmbedder | None = None,
        dense_searcher: DenseSearcher | None = None,
        sparse_searcher: SparseSearcher | None = None,
    ) -> None:
        metadata_store = MetadataStore(settings.database_path)
        self._settings = settings
        self._query_embedder = query_embedder or QueryEmbedder.get_instance(
            settings.embedding_model, settings.embedding_batch_size, settings.retrieval_cache_size
        )
        self._dense_searcher = dense_searcher or DenseSearcher(
            settings.vector_index_path, settings.database_path, metadata_store
        )
        self._sparse_searcher = sparse_searcher or SparseSearcher(metadata_store, settings.database_path)
        self._result_cache: LRUCache[str, tuple[HybridResult, ...]] = LRUCache(settings.retrieval_cache_size)

    def search(self, request: SearchRequest | str, limit: int | None = None) -> list[HybridResult]:
        """Retrieve the best document chunks using FAISS, BM25, and RRF."""
        normalized_request = (
            request if isinstance(request, SearchRequest) else SearchRequest(query=request, limit=limit or 5)
        )
        cache_key = normalized_request.model_dump_json()
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        total_start = perf_counter()
        logger.bind(event="retrieval_started").info("Hybrid retrieval started")
        embedding_start = perf_counter()
        query_vector = self._query_embedder.embed(normalized_request.query)
        logger.bind(event="embedding_completed").info("Query embedding completed in {:.3f}s", perf_counter() - embedding_start)

        dense_start = perf_counter()
        dense_results = self._dense_searcher.search(
            query_vector, self._settings.dense_candidates, normalized_request.filters
        )
        logger.bind(event="faiss_search_completed").info("FAISS search completed in {:.3f}s", perf_counter() - dense_start)

        sparse_start = perf_counter()
        sparse_results = self._sparse_searcher.search(
            normalized_request.query, self._settings.sparse_candidates, normalized_request.filters
        )
        logger.bind(event="bm25_search_completed").info("BM25 search completed in {:.3f}s", perf_counter() - sparse_start)

        fusion_start = perf_counter()
        results = reciprocal_rank_fusion(dense_results, sparse_results, self._settings.rrf_k)[
            : normalized_request.limit
        ]
        logger.bind(event="fusion_completed").info("RRF fusion completed in {:.3f}s", perf_counter() - fusion_start)
        logger.bind(event="retrieval_completed").info("Hybrid retrieval completed in {:.3f}s", perf_counter() - total_start)
        self._result_cache.put(cache_key, tuple(results))
        return results

    def refresh(self) -> None:
        """Invalidate cached searches and reload indexes after corpus changes."""
        self._result_cache = LRUCache(self._settings.retrieval_cache_size)
        self._dense_searcher.refresh()
        self._sparse_searcher.refresh()
