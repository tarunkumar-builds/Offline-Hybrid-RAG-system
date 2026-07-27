"""Reusable, cached query embedding service."""

from threading import Lock

import numpy as np

from app.ingestion.embeddings import EmbeddingGenerator
from app.retrieval.cache import LRUCache
from app.utils.errors import EmbeddingError, RetrievalError


class QueryEmbedder:
    """Generate normalized query embeddings while sharing one model per configuration."""

    _instances: dict[tuple[str, int, int], "QueryEmbedder"] = {}
    _instances_lock = Lock()

    def __init__(self, model_name: str, batch_size: int, cache_size: int) -> None:
        self._generator = EmbeddingGenerator(model_name, batch_size)
        self._cache: LRUCache[str, np.ndarray] = LRUCache(cache_size)

    @classmethod
    def get_instance(cls, model_name: str, batch_size: int, cache_size: int) -> "QueryEmbedder":
        """Return the process-wide embedder for the supplied model configuration."""
        key = (model_name, batch_size, cache_size)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(model_name, batch_size, cache_size)
            return cls._instances[key]

    def embed(self, query: str) -> np.ndarray:
        """Return a normalized query vector, caching immutable copies by query text."""
        cached = self._cache.get(query)
        if cached is not None:
            return cached.copy()
        try:
            vector = self._generator.embed([query])
        except EmbeddingError as error:
            raise RetrievalError(f"Unable to embed retrieval query: {error}") from error
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise RetrievalError("Query embedder returned an invalid vector shape")
        self._cache.put(query, vector.copy())
        return vector
