"""SHA256-based duplicate detection for source PDFs."""

from pathlib import Path

from app.database import MetadataStore
from app.utils.hashing import sha256_file


class DuplicateDetector:
    """Identify documents with content already represented in metadata."""

    def __init__(self, metadata_store: MetadataStore) -> None:
        self._metadata_store = metadata_store

    def file_hash(self, source_path: Path) -> str:
        """Calculate the canonical content hash used as duplicate identity."""
        return sha256_file(source_path)

    def is_duplicate(self, file_hash: str) -> bool:
        """Return whether a document with this hash has already been indexed."""
        return self._metadata_store.has_document_hash(file_hash)
