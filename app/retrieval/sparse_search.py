"""SQLite-metadata-backed sparse keyword retrieval."""

from pathlib import Path

from app.database.metadata import MetadataStore
from app.retrieval.bm25_index import BM25Index
from app.retrieval.filters import apply_filters
from app.retrieval.models import SearchFilters, SearchResult
from app.utils.errors import RetrievalConfigurationError


class SparseSearcher:
    """Build and query a BM25 index from the Phase 1 metadata database."""

    def __init__(self, metadata_store: MetadataStore, database_path: Path) -> None:
        self._metadata_store = metadata_store
        self._database_path = database_path
        self._index: BM25Index | None = None

    def refresh(self) -> None:
        """Rebuild the BM25 corpus from all persisted chunk metadata."""
        if not self._database_path.is_file():
            raise RetrievalConfigurationError(f"SQLite metadata database is missing: {self._database_path}")
        self._index = BM25Index(self._metadata_store.get_all_chunks())

    def search(self, query: str, limit: int, filters: SearchFilters | None = None) -> list[SearchResult]:
        """Return ranked keyword matches with optional metadata filtering."""
        if self._index is None:
            self.refresh()
        if self._index is None or self._index.size == 0:
            return []
        search_limit = self._index.size if filters is not None else min(self._index.size, limit)
        candidates = [
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_number=chunk.chunk_number,
                text=chunk.text,
                source="sparse",
                bm25_score=score,
            )
            for chunk, score in self._index.search(query, search_limit)
        ]
        return apply_filters(candidates, filters)[:limit]
