"""In-memory BM25 index built from persisted SQLite chunk metadata."""

import re

from rank_bm25 import BM25Okapi

from app.models import StoredChunk


class BM25Index:
    """Tokenize corpus chunks and expose ranked BM25 scores."""

    _token_pattern = re.compile(r"[^\W_]+", re.UNICODE)

    def __init__(self, chunks: list[StoredChunk]) -> None:
        self._chunks = chunks
        self._tokenized_corpus = [self.tokenize(chunk.text) for chunk in chunks]
        self._token_sets = [set(tokens) for tokens in self._tokenized_corpus]
        self._vocabulary = {token for tokens in self._tokenized_corpus for token in tokens}
        self._index = BM25Okapi(self._tokenized_corpus) if chunks else None

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Lowercase text and remove punctuation for keyword retrieval."""
        return cls._token_pattern.findall(text.lower())

    @property
    def size(self) -> int:
        """Return the indexed chunk count."""
        return len(self._chunks)

    def search(self, query: str, limit: int) -> list[tuple[StoredChunk, float]]:
        """Return chunks ordered by descending BM25 relevance score."""
        query_tokens = self.tokenize(query)
        if self._index is None or not query_tokens or not (set(query_tokens) & self._vocabulary):
            return []
        scores = self._index.get_scores(query_tokens)
        query_token_set = set(query_tokens)
        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (bool(query_token_set & self._token_sets[index]), float(scores[index])),
            reverse=True,
        )[:limit]
        return [
            (self._chunks[int(index)], float(scores[index]))
            for index in ranked_indices
        ]
