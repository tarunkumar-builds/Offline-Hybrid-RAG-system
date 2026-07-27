"""PDF text extraction using PyMuPDF."""

from pathlib import Path

import fitz
from loguru import logger

from app.models import PageText
from app.utils.errors import PDFLoadError


class PDFLoader:
    """Load non-empty pages from a PDF document."""

    def load(self, file_path: Path) -> list[PageText]:
        """Extract page text while converting parser errors to domain errors."""
        if not file_path.is_file():
            raise PDFLoadError(f"PDF does not exist: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise PDFLoadError(f"Expected a PDF file: {file_path}")

        logger.info("Loading PDF: {}", file_path)
        try:
            with fitz.open(file_path) as document:
                pages = []
                for page_index, page in enumerate(document):
                    text = page.get_text("text").strip()
                    if text:
                        pages.append(
                            PageText(
                                document_name=file_path.name,
                                document_path=file_path.resolve(),
                                page_number=page_index + 1,
                                text=text,
                            )
                        )
        except (fitz.FileDataError, fitz.EmptyFileError, RuntimeError, OSError) as error:
            logger.exception("Unable to load PDF: {}", file_path)
            raise PDFLoadError(f"Unable to read PDF '{file_path}': {error}") from error

        if not pages:
            raise PDFLoadError(f"PDF contains no extractable text: {file_path}")
        logger.info("Loaded {} non-empty pages from {}", len(pages), file_path.name)
        return pages
