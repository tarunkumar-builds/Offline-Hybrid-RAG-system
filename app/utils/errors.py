"""Domain exceptions for ingestion failures."""


class IngestionError(Exception):
    """Base exception for a recoverable ingestion failure."""


class PDFLoadError(IngestionError):
    """Raised when a PDF cannot be opened or has no usable text."""


class EmbeddingError(IngestionError):
    """Raised when embeddings cannot be generated."""


class VectorStoreError(IngestionError):
    """Raised when FAISS operations fail."""


class MetadataError(IngestionError):
    """Raised when SQLite metadata operations fail."""


class RetrievalError(Exception):
    """Raised when the local hybrid retrieval engine cannot complete a search."""


class RetrievalConfigurationError(RetrievalError):
    """Raised when required retrieval artifacts are unavailable."""


class RerankingError(Exception):
    """Raised when cross-encoder reranking cannot complete."""


class RerankerConfigurationError(RerankingError):
    """Raised when reranker model or device configuration is invalid."""


class GenerationError(Exception):
    """Raised when the local answer generation workflow cannot complete."""


class OllamaConnectionError(GenerationError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaResponseError(GenerationError):
    """Raised when Ollama returns an invalid response or model error."""


class EvaluationError(Exception):
    """Raised when evaluation data, metrics, or reports are invalid."""
