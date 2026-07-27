"""Loguru configuration."""

import sys
from pathlib import Path

from loguru import logger


def configure_logging(log_level: str, log_directory: Path = Path("logs")) -> None:
    """Configure human-readable console logs and structured rotating file logs."""
    log_directory.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.configure(extra={"event": "application"})
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {extra[event]} | {message}",
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_directory / "application.jsonl",
        level=log_level.upper(),
        rotation="10 MB",
        retention="10 days",
        serialize=True,
        backtrace=False,
        diagnose=False,
    )
