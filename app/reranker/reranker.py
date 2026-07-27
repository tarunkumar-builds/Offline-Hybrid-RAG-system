"""Locally hosted BGE cross-encoder inference service."""

from collections.abc import Callable, Sequence
from threading import Lock

import numpy as np
from loguru import logger

from app.reranker.config import RerankerConfig
from app.reranker.models import RerankCandidate
from app.utils.errors import RerankerConfigurationError, RerankingError


class CrossEncoderReranker:
    """Score query-document pairs with one lazily loaded local CrossEncoder."""

    _instances: dict[tuple[str, str], "CrossEncoderReranker"] = {}
    _instances_lock = Lock()

    def __init__(
        self,
        config: RerankerConfig,
        model_loader: Callable[[str, str], object] | None = None,
    ) -> None:
        self._config = config
        self._model_loader = model_loader or self._load_model
        self._model: object | None = None

    @classmethod
    def get_instance(cls, config: RerankerConfig) -> "CrossEncoderReranker":
        """Return one shared reranker for each model and device combination."""
        key = (config.model_name, config.device)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(config)
            return cls._instances[key]

    def score(self, query: str, candidates: Sequence[RerankCandidate]) -> np.ndarray:
        """Run batched cross-encoder inference and return one score per candidate."""
        if not candidates:
            return np.empty(0, dtype=np.float32)
        pairs = [(query, candidate.text) for candidate in candidates]
        try:
            scores = self._get_model().predict(
                pairs, batch_size=self._config.batch_size, show_progress_bar=False
            )
        except RerankingError:
            raise
        except Exception as error:
            logger.exception("Cross-encoder inference failed")
            raise RerankingError(f"Cross-encoder inference failed: {error}") from error
        normalized_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if normalized_scores.size != len(candidates):
            raise RerankingError("Cross-encoder returned an unexpected number of scores")
        return normalized_scores

    def _get_model(self):
        if self._model is None:
            self._model = self._model_loader(self._config.model_name, self._config.device)
        return self._model

    @staticmethod
    def _load_model(model_name: str, device: str):
        try:
            import torch

            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RerankerConfigurationError("CUDA was requested but no GPU is available")
            from sentence_transformers import CrossEncoder

            logger.info("Loading local cross-encoder model {} on {}", model_name, device)
            return CrossEncoder(model_name, device=device, local_files_only=True)
        except RerankerConfigurationError:
            raise
        except Exception as error:
            raise RerankerConfigurationError(
                f"Unable to load local reranker model '{model_name}': {error}"
            ) from error
