from pathlib import Path

from app.ingestion.chunker import TokenChunker
from app.models import PageText


def test_chunker_creates_ordered_overlapping_chunks() -> None:
    text = " ".join(f"Sentence {number}." for number in range(1, 15))
    page = PageText(document_name="doc.pdf", document_path=Path("doc.pdf"), page_number=1, text=text)

    chunks = TokenChunker(chunk_size=12, chunk_overlap=4).chunk_pages([page], "document-id")

    assert len(chunks) > 1
    assert [chunk.chunk_number for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.document_id == "document-id" for chunk in chunks)
    assert "Sentence 5." in chunks[1].text
