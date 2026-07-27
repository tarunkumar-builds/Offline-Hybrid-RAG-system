"""SQLite metadata store for documents and chunks."""

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.models import Chunk, DocumentRecord, StoredChunk
from app.utils.errors import MetadataError


class MetadataStore:
    """Persist document and chunk records in a local SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        """Create the schema and indexes if they are not already present."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    PRAGMA foreign_keys = ON;
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        document_name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_hash TEXT NOT NULL UNIQUE,
                        ingestion_time TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        page_number INTEGER NOT NULL,
                        chunk_number INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        embedding_dimension INTEGER NOT NULL,
                        vector_id INTEGER NOT NULL UNIQUE,
                        ingestion_time TEXT NOT NULL,
                        FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
                        UNIQUE(document_id, chunk_number)
                    );
                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                    """
                )
            logger.info("Metadata database initialized: {}", self._database_path)
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to initialize SQLite database: {error}") from error

    def has_document_hash(self, file_hash: str) -> bool:
        """Return whether this exact file content was indexed previously."""
        try:
            with self._connect() as connection:
                return connection.execute(
                    "SELECT 1 FROM documents WHERE file_hash = ? LIMIT 1", (file_hash,)
                ).fetchone() is not None
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to check document duplicate: {error}") from error

    def add_document(
        self,
        document: DocumentRecord,
        chunks: Sequence[Chunk],
        vector_ids: Sequence[int],
        embedding_dimension: int,
    ) -> None:
        """Atomically store a document and all corresponding chunk metadata."""
        if len(chunks) != len(vector_ids):
            raise MetadataError("Each chunk requires exactly one vector ID")
        timestamp = document.ingestion_time.astimezone(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                    (
                        document.document_id,
                        document.document_name,
                        str(document.file_path),
                        document.file_hash,
                        timestamp,
                    ),
                )
                connection.executemany(
                    """INSERT INTO chunks (
                        chunk_id, document_id, page_number, chunk_number, text,
                        embedding_dimension, vector_id, ingestion_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.page_number,
                            chunk.chunk_number,
                            chunk.text,
                            embedding_dimension,
                            vector_id,
                            timestamp,
                        )
                        for chunk, vector_id in zip(chunks, vector_ids, strict=True)
                    ],
                )
            logger.info("Stored {} chunk records for {}", len(chunks), document.document_name)
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to store metadata: {error}") from error

    def remove_document(self, document_id: str) -> None:
        """Remove metadata after a downstream write failure."""
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to remove metadata: {error}") from error

    def list_documents(self) -> list[DocumentRecord]:
        """Return all indexed documents in ingestion order."""
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT document_id, document_name, file_path, file_hash, ingestion_time FROM documents ORDER BY ingestion_time"
                ).fetchall()
            return [
                DocumentRecord(
                    document_id=row[0], document_name=row[1], file_path=Path(row[2]), file_hash=row[3], ingestion_time=row[4]
                )
                for row in rows
            ]
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to list documents: {error}") from error

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one indexed document, if present."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT document_id, document_name, file_path, file_hash, ingestion_time FROM documents WHERE document_id = ?",
                    (document_id,),
                ).fetchone()
            return (
                DocumentRecord(document_id=row[0], document_name=row[1], file_path=Path(row[2]), file_hash=row[3], ingestion_time=row[4])
                if row else None
            )
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to get document: {error}") from error

    def get_document_by_hash(self, file_hash: str) -> DocumentRecord | None:
        """Return the indexed document with the supplied content hash, if any."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT document_id, document_name, file_path, file_hash, ingestion_time FROM documents WHERE file_hash = ?",
                    (file_hash,),
                ).fetchone()
            return (
                DocumentRecord(document_id=row[0], document_name=row[1], file_path=Path(row[2]), file_hash=row[3], ingestion_time=row[4])
                if row else None
            )
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to get document by hash: {error}") from error

    def get_document_vector_data(self, document_id: str) -> tuple[list[int], int | None]:
        """Return vector IDs and their embedding dimension before document removal."""
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT vector_id, embedding_dimension FROM chunks WHERE document_id = ?", (document_id,)
                ).fetchall()
            return [int(row[0]) for row in rows], (int(rows[0][1]) if rows else None)
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to read document vectors: {error}") from error

    def chunk_count(self) -> int:
        """Return the number of stored chunk records for monitoring and tests."""
        try:
            with self._connect() as connection:
                return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to count chunk metadata: {error}") from error

    def get_document_chunk_counts(self) -> dict[str, int]:
        """Return indexed chunk totals keyed by document ID for API summaries."""
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT document_id, COUNT(*) FROM chunks GROUP BY document_id"
                ).fetchall()
            return {str(document_id): int(count) for document_id, count in rows}
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to count document chunks: {error}") from error

    def get_all_chunks(self) -> list[StoredChunk]:
        """Return every indexed chunk with document data for retrieval indexing."""
        return list(self._read_chunks())

    def get_chunks_by_vector_ids(self, vector_ids: Sequence[int]) -> dict[int, StoredChunk]:
        """Return chunk metadata keyed by FAISS vector identifier."""
        if not vector_ids:
            return {}
        placeholders = ", ".join("?" for _ in vector_ids)
        query = f"{self._chunk_select()} WHERE c.vector_id IN ({placeholders})"
        return {chunk.vector_id: chunk for chunk in self._read_chunks(query, tuple(vector_ids))}

    def _read_chunks(
        self,
        query: str | None = None,
        parameters: tuple[int, ...] = (),
    ) -> list[StoredChunk]:
        try:
            with self._connect() as connection:
                rows = connection.execute(query or self._chunk_select(), parameters).fetchall()
            return [
                StoredChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    document_name=row[2],
                    page_number=row[3],
                    chunk_number=row[4],
                    text=row[5],
                    vector_id=row[6],
                )
                for row in rows
            ]
        except sqlite3.Error as error:
            raise MetadataError(f"Unable to read chunk metadata: {error}") from error

    @staticmethod
    def _chunk_select() -> str:
        return """
            SELECT c.chunk_id, c.document_id, d.document_name, c.page_number,
                   c.chunk_number, c.text, c.vector_id
            FROM chunks AS c
            INNER JOIN documents AS d ON d.document_id = c.document_id
        """

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
