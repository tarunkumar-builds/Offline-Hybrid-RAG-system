from pathlib import Path

import fitz
import pytest

from app.ingestion.pdf_loader import PDFLoader
from app.utils.errors import PDFLoadError


def test_pdf_loader_extracts_non_empty_pages(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "First page text")
    document.new_page()
    document.save(file_path)
    document.close()

    pages = PDFLoader().load(file_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].document_name == "sample.pdf"
    assert "First page text" in pages[0].text


def test_pdf_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PDFLoadError, match="does not exist"):
        PDFLoader().load(tmp_path / "missing.pdf")
