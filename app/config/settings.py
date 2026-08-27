"""Typed configuration for the offline ingestion pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or an optional `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = Field(default=450, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)
    documents_dir: Path = Path("documents")
    vector_index_path: Path = Path("vector_store/documents.faiss")
    database_path: Path = Path("metadata/metadata.db")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    max_upload_files: int = Field(default=10, ge=1, le=100)
    max_upload_size_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1)
    dense_candidates: int = Field(default=20, ge=1)
    sparse_candidates: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    retrieval_cache_size: int = Field(default=128, ge=1)
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_k_input: int = Field(default=20, ge=1)
    reranker_top_n_output: int = Field(default=5, ge=1)
    reranker_batch_size: int = Field(default=16, ge=1)
    reranker_device: str = "cpu"
    reranker_cache_size: int = Field(default=128, ge=1)
    ollama_model: str = "qwen3:latest"
    ollama_base_url: str = "http://127.0.0.1:11434"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    generation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    generation_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    generation_max_tokens: int = Field(default=512, ge=1)
    generation_timeout_seconds: float = Field(default=60.0, gt=0.0)
    generation_prompt_template: str = "default"
    generation_streaming: bool = False
    generation_retries: int = Field(default=2, ge=0)
    evaluation_output_dir: Path = Path("reports")
    evaluation_report_format: str = "json"
    evaluation_dataset_path: Path | None = None
    evaluation_citation_coverage_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        """Ensure overlap leaves room for new content."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.reranker_top_n_output > self.reranker_top_k_input:
            raise ValueError("reranker_top_n_output cannot exceed reranker_top_k_input")
        if not self.ollama_model.strip():
            raise ValueError("ollama_model must not be blank")
        if self.evaluation_report_format not in {"json", "csv", "markdown"}:
            raise ValueError("evaluation_report_format must be json, csv, or markdown")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        """Return configured frontend origins as a normalized list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def ensure_directories(self) -> None:
        """Create local data directories required by the pipeline."""
        for directory in (
            self.documents_dir / "raw",
            self.documents_dir / "processed",
            self.vector_index_path.parent,
            self.database_path.parent,
            self.evaluation_output_dir,
            Path("logs"),
        ):
            directory.mkdir(parents=True, exist_ok=True)
