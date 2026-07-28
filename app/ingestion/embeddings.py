"""Local sentence-transformer embedding generation."""

from collections.abc import Sequence

import numpy as np
from loguru import logger

from app.utils.errors import EmbeddingError


class EmbeddingGenerator:
    """Lazily load and use a local Sentence Transformers model."""

    def __init__(self, model_name: str, batch_size: int) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension after loading the model."""
        return int(self._get_model().get_sentence_embedding_dimension())

    def count_tokens(self, text: str) -> int:
        """Count model tokenizer tokens without adding special tokens."""
        tokenizer = self._get_model().tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed text locally and L2-normalize every vector."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            logger.info("Generating embeddings for {} chunks", len(texts))
            vectors = self._get_model().encode(
                list(texts),
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            logger.info(
                "Embeddings generated successfully. Shape={}, dtype={}",
                vectors.shape,
                vectors.dtype,
            )

            return np.ascontiguousarray(vectors, dtype=np.float32)
        except Exception as error:  # Model backends expose heterogeneous exception types.
            logger.exception("Embedding generation failed")
            raise EmbeddingError(f"Failed to generate embeddings: {error}") from error

    def _get_model(self):
        if self._model is None:
            try:

                from sentence_transformers import SentenceTransformer
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"

                logger.info(
                    "Loading embedding model: {} on {}",
                    self._model_name,
                    device,
                )

                self._model = SentenceTransformer(
                    self._model_name,
                    device=device,
                    local_files_only=True,
                )
            except Exception as error:
                raise EmbeddingError(
                    f"Unable to load local model '{self._model_name}'. Download it before offline use: {error}"
                ) from error
        return self._model
