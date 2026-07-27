"""Persistent FAISS exact inner-product index."""

from pathlib import Path

import numpy as np
from loguru import logger

from app.utils.errors import VectorStoreError


class FaissIndexManager:
    """Create, persist, and append to an ID-mapped `IndexFlatIP` index."""

    def __init__(self, index_path: Path, dimension: int) -> None:
        self._index_path = index_path
        self._dimension = dimension
        self._index = None

    @property
    def size(self) -> int:
        """Return the number of vectors currently indexed."""
        return int(self._get_index().ntotal)

    def add(self, vectors: np.ndarray, vector_ids: np.ndarray) -> None:
        """Append normalized vectors paired with stable integer IDs."""
        if vectors.ndim != 2 or vectors.shape[1] != self._dimension:
            raise VectorStoreError(f"Expected vectors of shape (n, {self._dimension})")
        if len(vectors) != len(vector_ids):
            raise VectorStoreError("Vector count does not match vector ID count")
        try:
            self._get_index().add_with_ids(
                np.ascontiguousarray(vectors, dtype=np.float32),
                np.ascontiguousarray(vector_ids, dtype=np.int64),
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to append FAISS vectors: {error}") from error

    def save(self) -> None:
        """Write the index atomically so existing index files remain intact."""
        try:
            import faiss

            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
            faiss.write_index(self._get_index(), str(temporary_path))
            temporary_path.replace(self._index_path)
            logger.info("Saved FAISS index with {} vectors to {}", self.size, self._index_path)
        except Exception as error:
            raise VectorStoreError(f"Unable to save FAISS index: {error}") from error

    def search(self, query_vectors: np.ndarray, limit: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search placeholder returning inner-product scores and vector IDs."""
        try:
            return self._get_index().search(np.ascontiguousarray(query_vectors, dtype=np.float32), limit)
        except Exception as error:
            raise VectorStoreError(f"Unable to search FAISS index: {error}") from error

    def remove(self, vector_ids: np.ndarray) -> int:
        """Remove vectors by stable IDs and return the number removed."""
        try:
            return int(self._get_index().remove_ids(np.ascontiguousarray(vector_ids, dtype=np.int64)))
        except Exception as error:
            raise VectorStoreError(f"Unable to remove FAISS vectors: {error}") from error

    def _get_index(self):
        if self._index is None:
            try:
                import faiss

                if self._index_path.exists():
                    index = faiss.read_index(str(self._index_path))
                    if index.d != self._dimension:
                        raise VectorStoreError(
                            f"FAISS index dimension {index.d} differs from model dimension {self._dimension}"
                        )
                    self._index = index
                    logger.info("Loaded FAISS index with {} vectors", index.ntotal)
                else:
                    self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
                    logger.info("Created FAISS IndexFlatIP with dimension {}", self._dimension)
            except VectorStoreError:
                raise
            except Exception as error:
                raise VectorStoreError(f"Unable to load FAISS index: {error}") from error
        return self._index
