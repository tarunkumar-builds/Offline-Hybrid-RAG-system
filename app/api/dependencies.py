"""Singleton service wiring and API-facing orchestration."""

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import RLock

import httpx
import numpy as np
from loguru import logger

from app.config import Settings
from app.database import MetadataStore
from app.evaluation import BenchmarkRunner, DatasetRecord, EvaluationInput, EvaluationPipeline
from app.generation import GenerationPipeline
from app.ingestion.faiss_index import FaissIndexManager
from app.ingestion.pipeline import IngestionPipeline
from app.utils.hashing import sha256_file


class ApiServices:
    """Service facade shared by all API endpoints."""

    _MAX_PIPELINE_VARIANTS = 8

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.metadata = MetadataStore(settings.database_path)
        self.metadata.initialize()
        self.ingestion = IngestionPipeline(settings)
        self.evaluation = EvaluationPipeline.from_settings(settings)
        self._ollama_health_client = httpx.Client(timeout=2.0)
        self._pipeline_lock = RLock()
        self._mutation_lock = RLock()
        self._generation_pipelines: OrderedDict[tuple[int, str, float, str], GenerationPipeline] = OrderedDict()

    def list_documents(self):
        return self.metadata.list_documents() if self.settings.database_path.exists() else []

    def get_document(self, document_id: str):
        return self.metadata.get_document(document_id) if self.settings.database_path.exists() else None

    def document_chunk_counts(self) -> dict[str, int]:
        """Return chunk totals for the current indexed document collection."""
        return self.metadata.get_document_chunk_counts() if self.settings.database_path.exists() else {}

    def ingest_document(self, source_path: Path):
        """Ingest a validated PDF and refresh query state after a corpus mutation."""
        with self._mutation_lock:
            file_hash = sha256_file(source_path)
            if self.metadata.has_document_hash(file_hash):
                return None
            indexed_chunks = self.ingestion.ingest(source_path)
            document = self.metadata.get_document_by_hash(file_hash)
            if document is None:
                raise RuntimeError("Ingestion completed without a stored document")
            self._refresh_generation_pipelines()
            logger.bind(event="document_uploaded").info("Indexed document with {} chunks", indexed_chunks)
            return document, indexed_chunks

    def delete_document(self, document_id: str) -> bool:
        """Delete one document and invalidate cached retrieval results."""
        with self._mutation_lock:
            document = self.get_document(document_id)
            if document is None:
                return False
            vector_ids, dimension = self.metadata.get_document_vector_data(document_id)
            if vector_ids and dimension and self.settings.vector_index_path.exists():
                index = FaissIndexManager(self.settings.vector_index_path, dimension)
                index.remove(np.asarray(vector_ids, dtype=np.int64))
                index.save()
            self.metadata.remove_document(document_id)
            self.ingestion.refresh_index()
            self._delete_owned_file(document.file_path, self.settings.documents_dir / "raw")
            self._delete_owned_file(
                self.settings.documents_dir / "processed" / f"{document_id}.json",
                self.settings.documents_dir / "processed",
            )
            self._refresh_generation_pipelines()
            logger.bind(event="document_deleted").info("Deleted indexed document")
            return True

    def answer(
        self,
        question: str,
        top_k: int,
        model_name: str | None,
        temperature: float | None,
        template: str | None,
    ):
        """Run a cached generation pipeline configured for the request controls."""
        pipeline = self._get_generation_pipeline(top_k, model_name, temperature, template)
        return pipeline.answer(question)

    def _get_generation_pipeline(
        self,
        top_k: int,
        model_name: str | None,
        temperature: float | None,
        template: str | None,
    ) -> GenerationPipeline:
        resolved_model = model_name or self.settings.ollama_model
        resolved_temperature = temperature if temperature is not None else self.settings.generation_temperature
        resolved_template = template or self.settings.generation_prompt_template
        key = (top_k, resolved_model, resolved_temperature, resolved_template)
        with self._pipeline_lock:
            existing = self._generation_pipelines.pop(key, None)
            if existing is not None:
                self._generation_pipelines[key] = existing
                return existing

            overrides: dict[str, object] = self.settings.model_dump()
            overrides.update(
                {
                    "reranker_top_k_input": max(top_k, self.settings.reranker_top_k_input),
                    "reranker_top_n_output": top_k,
                    "ollama_model": resolved_model,
                    "generation_temperature": resolved_temperature,
                    "generation_prompt_template": resolved_template,
                }
            )
            pipeline = GenerationPipeline.from_settings(Settings(**overrides))
            if len(self._generation_pipelines) >= self._MAX_PIPELINE_VARIANTS:
                _, evicted = self._generation_pipelines.popitem(last=False)
                evicted.close()
            self._generation_pipelines[key] = pipeline
            return pipeline

    def _refresh_generation_pipelines(self) -> None:
        with self._pipeline_lock:
            for pipeline in self._generation_pipelines.values():
                pipeline.refresh()

    @staticmethod
    def _delete_owned_file(path: Path, parent: Path) -> None:
        resolved_path = path.resolve()
        resolved_parent = parent.resolve()
        if resolved_path.is_file() and resolved_path.is_relative_to(resolved_parent):
            resolved_path.unlink()

    def benchmark(self, records: list[DatasetRecord]):
        """Execute a benchmark using the same cached query pipelines as the API."""
        def execute(record: DatasetRecord) -> EvaluationInput:
            answer = self.answer(record.question, self.settings.reranker_top_n_output, None, None, None)
            return EvaluationInput(
                question=record.question,
                reranked_chunks=answer.retrieved_chunks,
                generated_answer=answer,
            )

        return BenchmarkRunner().run(records, execute)

    def public_config(self) -> dict[str, object]:
        """Return non-sensitive configuration suitable for the public API."""
        return {
            "embedding_model": self.settings.embedding_model,
            "reranker_model": self.settings.reranker_model,
            "ollama_model": self.settings.ollama_model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "retrieval": {
                "dense_candidates": self.settings.dense_candidates,
                "sparse_candidates": self.settings.sparse_candidates,
                "reranker_top_k_input": self.settings.reranker_top_k_input,
                "reranker_top_n_output": self.settings.reranker_top_n_output,
            },
            "generation": {
                "temperature": self.settings.generation_temperature,
                "max_tokens": self.settings.generation_max_tokens,
                "prompt_template": self.settings.generation_prompt_template,
            },
            "evaluation": {"report_format": self.settings.evaluation_report_format},
        }

    def health(self) -> tuple[str, int, int, str]:
        database_status = "ready" if self.settings.database_path.exists() else "missing"
        document_count = len(self.list_documents())
        vector_count = self.metadata.chunk_count() if self.settings.database_path.exists() else 0
        try:
            response = self._ollama_health_client.get(f"{self.settings.ollama_base_url.rstrip('/')}/api/tags")
            ollama_status = "ready" if response.is_success else "unavailable"
        except httpx.HTTPError:
            ollama_status = "unavailable"
        return ollama_status, document_count, vector_count, database_status

    def close(self) -> None:
        """Close process-wide HTTP clients when the application shuts down."""
        with self._pipeline_lock:
            for pipeline in self._generation_pipelines.values():
                pipeline.close()
            self._generation_pipelines.clear()
        self._ollama_health_client.close()


@lru_cache
def get_settings() -> Settings:
    """Return the shared environment-backed settings instance."""
    return Settings()


@lru_cache
def get_services() -> ApiServices:
    """Return singleton application services for dependency injection."""
    return ApiServices(get_settings())
