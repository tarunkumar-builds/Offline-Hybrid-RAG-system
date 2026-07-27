"""Evaluation API contracts."""

from pathlib import PurePath, PurePosixPath, PureWindowsPath

from pydantic import BaseModel, Field, field_validator

from app.evaluation.models import BenchmarkResult, EvaluationInput, EvaluationResult


class EvaluationQueryRequest(BaseModel):
    """A completed RAG execution to score and optionally report."""

    evaluation_input: EvaluationInput
    report_name: str = Field(default="evaluation", min_length=1, max_length=80)

    @field_validator("report_name")
    @classmethod
    def validate_report_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or PurePath(normalized).name != normalized:
            raise ValueError("report_name must be a simple filename")
        if not all(character.isalnum() or character in {"-", "_"} for character in normalized):
            raise ValueError("report_name may contain only letters, numbers, hyphens, and underscores")
        return normalized


class BenchmarkRequest(BaseModel):
    """Relative path to a local benchmark dataset, when configured data is not used."""

    dataset_path: str | None = Field(default=None, max_length=512)

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        paths = (PurePath(normalized), PurePosixPath(normalized), PureWindowsPath(normalized))
        if not normalized or any(path.is_absolute() or ".." in path.parts for path in paths):
            raise ValueError("dataset_path must be a relative path within the application directory")
        return normalized


class EvaluationResponse(BaseModel):
    result: EvaluationResult
    processing_time: float = Field(ge=0.0)


class BenchmarkResponse(BenchmarkResult):
    """Benchmark metrics with overall API processing duration."""

    processing_time: float = Field(ge=0.0)
