"""Orchestrator for offline, incremental PDF ingestion."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from loguru import logger

from app.config import Settings
from app.database import MetadataStore
from app.ingestion.chunker import TokenChunker
from app.ingestion.duplicate_detector import DuplicateDetector
from app.ingestion.embeddings import EmbeddingGenerator
from app.ingestion.faiss_index import FaissIndexManager
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_cleaner import TextCleaner
from app.models import Chunk, DocumentRecord
from app.utils.errors import IngestionError


class IngestionPipeline:
    """Coordinate file validation, extraction, embeddings, and durable storage."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loader = PDFLoader()
        self._cleaner = TextCleaner()
        self._chunker = TokenChunker(settings.chunk_size, settings.chunk_overlap)
        self._embedder = EmbeddingGenerator(settings.embedding_model, settings.embedding_batch_size)
        self._metadata = MetadataStore(settings.database_path)
        self._duplicates = DuplicateDetector(self._metadata)
        self._index: FaissIndexManager | None = None

    def ingest(self, source_path: Path) -> int:
        """Ingest one PDF, returning the number of newly indexed chunks."""
        source_path = source_path.expanduser().resolve()
        self._settings.ensure_directories()
        self._metadata.initialize()
        file_hash = self._duplicates.file_hash(source_path) if source_path.is_file() else ""
        if self._duplicates.is_duplicate(file_hash):
            logger.info("Skipping duplicate PDF: {}", source_path)
            return 0

        pages = self._loader.load(source_path)
        cleaned_pages = [page.model_copy(update={"text": self._cleaner.clean(page.text)}) for page in pages]
        cleaned_pages = [page for page in cleaned_pages if page.text]
        if not cleaned_pages:
            raise IngestionError(f"PDF has no usable text after cleaning: {source_path}")

        document_id = str(uuid5(NAMESPACE_URL, file_hash))
        chunks = self._chunker.chunk_pages(cleaned_pages, document_id, self._embedder.count_tokens)
        if not chunks:
            raise IngestionError(f"No chunks were created for: {source_path}")
        logger.info("Created {} chunks for {}", len(chunks), source_path.name)

        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        index = self._get_index()
        vector_ids = self._vector_ids(chunks)
        stored_path = self._copy_original(source_path, file_hash)
        document = DocumentRecord(
            document_id=document_id,
            document_name=source_path.name,
            file_path=stored_path,
            file_hash=file_hash,
            ingestion_time=datetime.now(timezone.utc),
        )

        self._metadata.add_document(document, chunks, vector_ids.tolist(), vectors.shape[1])
        try:
            index.add(vectors, vector_ids)
            index.save()
        except Exception:
            self._metadata.remove_document(document_id)
            raise
        self._save_processed_copy(document, chunks)
        logger.info("Indexed {} chunks from {}", len(chunks), source_path.name)
        return len(chunks)

    def ingest_path(self, path: Path) -> dict[Path, int]:
        """Ingest a single PDF or all PDFs beneath a directory."""
        if path.is_file():
            return {path: self.ingest(path)}
        if not path.is_dir():
            raise IngestionError(f"Input path does not exist: {path}")
        results: dict[Path, int] = {}
        for pdf_path in sorted(path.rglob("*.pdf")):
            results[pdf_path] = self.ingest(pdf_path)
        return results

    def refresh_index(self) -> None:
        """Reload the persisted FAISS index after an external corpus mutation."""
        self._index = None

    def _get_index(self) -> FaissIndexManager:
        if self._index is None:
            self._index = FaissIndexManager(self._settings.vector_index_path, self._embedder.dimension)
        return self._index

    def _copy_original(self, source_path: Path, file_hash: str) -> Path:
        destination = self._settings.documents_dir / "raw" / f"{file_hash[:12]}_{source_path.name}"
        if not destination.exists():
            shutil.copy2(source_path, destination)
        return destination.resolve()

    def _save_processed_copy(self, document: DocumentRecord, chunks: list[Chunk]) -> None:
        payload = {
            "document_id": document.document_id,
            "document_name": document.document_name,
            "file_hash": document.file_hash,
            "chunks": [chunk.model_dump() for chunk in chunks],
        }
        processed_path = self._settings.documents_dir / "processed" / f"{document.document_id}.json"
        processed_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _vector_ids(chunks: list[Chunk]) -> np.ndarray:
        values = [int(uuid5(NAMESPACE_URL, chunk.chunk_id).int % (2**63 - 1)) for chunk in chunks]
        if len(values) != len(set(values)):
            raise IngestionError("Generated colliding FAISS vector IDs")
        return np.asarray(values, dtype=np.int64)
