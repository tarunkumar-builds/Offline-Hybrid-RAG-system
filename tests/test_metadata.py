from datetime import datetime, timezone
from pathlib import Path

from app.database.metadata import MetadataStore
from app.models import Chunk, DocumentRecord


def test_metadata_store_persists_chunks_and_detects_duplicates(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.db")
    store.initialize()
    document = DocumentRecord(
        document_id="document-1",
        document_name="sample.pdf",
        file_path=tmp_path / "sample.pdf",
        file_hash="a" * 64,
        ingestion_time=datetime.now(timezone.utc),
    )
    chunk = Chunk(chunk_id="chunk-1", document_id="document-1", page_number=1, chunk_number=0, text="Text")

    store.add_document(document, [chunk], [99], embedding_dimension=384)

    assert store.has_document_hash("a" * 64)
    assert store.chunk_count() == 1
