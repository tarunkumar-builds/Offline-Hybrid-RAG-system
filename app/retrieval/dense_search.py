"""FAISS-backed dense retrieval over Phase 1 vectors."""

from pathlib import Path

import numpy as np

from app.database.metadata import MetadataStore
from app.ingestion.faiss_index import FaissIndexManager
from app.retrieval.filters import apply_filters
from app.retrieval.models import SearchFilters, SearchResult
from app.utils.errors import RetrievalConfigurationError, RetrievalError, VectorStoreError


class DenseSearcher:
    """Look up normalized query vectors in the persisted FAISS index."""

    def __init__(self, index_path: Path, database_path: Path, metadata_store: MetadataStore) -> None:
        self._index_path = index_path
        self._database_path = database_path
        self._metadata_store = metadata_store
        self._manager: FaissIndexManager | None = None

    def search(
        self,
        query_vector: np.ndarray,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Return highest cosine-similarity chunks, subject to metadata filters."""
        if not self._index_path.is_file():
            raise RetrievalConfigurationError(f"FAISS index is missing: {self._index_path}")
        if not self._database_path.is_file():
            raise RetrievalConfigurationError(f"SQLite metadata database is missing: {self._database_path}")
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise RetrievalError("Dense search requires one query vector")
        try:
            if self._manager is None:
                self._manager = FaissIndexManager(self._index_path, int(query_vector.shape[1]))
            search_limit = self._manager.size if filters is not None else min(self._manager.size, limit)
            if search_limit == 0:
                return []
            scores, vector_ids = self._manager.search(query_vector, search_limit)
        except VectorStoreError as error:
            raise RetrievalError(f"FAISS search failed: {error}") from error

        valid_ids = [int(vector_id) for vector_id in vector_ids[0] if vector_id >= 0]
        metadata = self._metadata_store.get_chunks_by_vector_ids(valid_ids)
        candidates = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_number=chunk.chunk_number,
                text=chunk.text,
                source="dense",
                similarity_score=float(score),
            )
            for score, vector_id in zip(scores[0], vector_ids[0], strict=True)
            if vector_id >= 0 and (chunk := metadata.get(int(vector_id))) is not None
        ]
        return apply_filters(candidates, filters)[:limit]

    def refresh(self) -> None:
        """Reload the FAISS index on the next search after a corpus mutation."""
        self._manager = None
