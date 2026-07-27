"""Configuration for output and validation behavior of evaluations."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import Settings


class EvaluationConfig(BaseModel):
    """Runtime options for reports, datasets, and metric thresholds."""

    model_config = ConfigDict(frozen=True)

    output_directory: Path = Path("reports")
    report_format: Literal["json", "csv", "markdown"] = "json"
    dataset_path: Path | None = None
    citation_coverage_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @classmethod
    def from_settings(cls, settings: Settings) -> "EvaluationConfig":
        """Create evaluation configuration from shared environment settings."""
        return cls(
            output_directory=settings.evaluation_output_dir,
            report_format=settings.evaluation_report_format,
            dataset_path=settings.evaluation_dataset_path,
            citation_coverage_threshold=settings.evaluation_citation_coverage_threshold,
        )
