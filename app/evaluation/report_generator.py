"""JSON, CSV, and Markdown report generation for evaluation results."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from app.evaluation.models import BenchmarkResult, EvaluationResult
from app.utils.errors import EvaluationError


class ReportGenerator:
    """Persist evaluation output in portable local formats."""

    def __init__(self, output_directory: Path) -> None:
        self._output_directory = output_directory

    def write_evaluation(self, result: EvaluationResult, report_format: str, name: str = "evaluation") -> Path:
        """Write a single-query report in the requested format."""
        return self._write(result.model_dump(mode="json"), report_format, name)

    def write_benchmark(self, result: BenchmarkResult, report_format: str, name: str = "benchmark") -> Path:
        """Write an aggregate benchmark report in the requested format."""
        return self._write(result.model_dump(mode="json"), report_format, name)

    def _write(self, payload: dict[str, object], report_format: str, name: str) -> Path:
        self._output_directory.mkdir(parents=True, exist_ok=True)
        if report_format == "json":
            path = self._output_directory / f"{name}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elif report_format == "csv":
            path = self._output_directory / f"{name}.csv"
            pd.json_normalize(payload).to_csv(path, index=False)
        elif report_format == "markdown":
            path = self._output_directory / f"{name}.md"
            path.write_text(self._markdown(payload), encoding="utf-8")
        else:
            raise EvaluationError("Report format must be json, csv, or markdown")
        logger.info("Evaluation report generated: {}", path)
        return path

    @staticmethod
    def _markdown(payload: dict[str, object]) -> str:
        lines = ["# RAG Evaluation Report", ""]
        for key, value in payload.items():
            lines.extend([f"## {key.replace('_', ' ').title()}", ""])
            if isinstance(value, dict):
                lines.extend(f"- **{metric.replace('_', ' ')}:** {item}" for metric, item in value.items())
            elif isinstance(value, list):
                lines.append(f"- Entries: {len(value)}")
            else:
                lines.append(str(value))
            lines.append("")
        return "\n".join(lines)
