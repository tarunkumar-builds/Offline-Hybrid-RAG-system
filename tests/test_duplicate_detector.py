from pathlib import Path

from app.database.metadata import MetadataStore
from app.ingestion.duplicate_detector import DuplicateDetector


def test_duplicate_detector_hashes_pdf_content(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"same content")
    store = MetadataStore(tmp_path / "metadata.db")
    store.initialize()

    detector = DuplicateDetector(store)

    assert len(detector.file_hash(source)) == 64
    assert not detector.is_duplicate(detector.file_hash(source))
