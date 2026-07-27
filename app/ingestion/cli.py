"""Command-line entry point for document ingestion."""

import argparse
from pathlib import Path

from loguru import logger

from app.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.utils.errors import IngestionError
from app.utils.logging import configure_logging


def main() -> int:
    """Run ingestion for a PDF file or directory of PDFs."""
    parser = argparse.ArgumentParser(description="Index PDFs into the local offline RAG store.")
    parser.add_argument("path", type=Path, help="PDF file or directory containing PDFs")
    arguments = parser.parse_args()
    settings = Settings()
    configure_logging(settings.log_level)
    try:
        results = IngestionPipeline(settings).ingest_path(arguments.path)
    except IngestionError as error:
        logger.error("Ingestion failed: {}", error)
        return 1
    indexed = sum(1 for chunk_count in results.values() if chunk_count)
    skipped = sum(1 for chunk_count in results.values() if not chunk_count)
    logger.info("Completed ingestion: {} indexed, {} duplicates skipped", indexed, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
