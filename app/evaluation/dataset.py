"""Local benchmark dataset loading from JSON, YAML, or CSV."""

import json
from pathlib import Path

import pandas as pd
import yaml

from app.evaluation.models import DatasetRecord
from app.utils.errors import EvaluationError


class DatasetLoader:
    """Load validated benchmark records without any network dependency."""

    def load(self, path: Path) -> list[DatasetRecord]:
        """Read JSON, YAML, or CSV dataset rows and validate their shape."""
        if not path.is_file():
            raise EvaluationError(f"Benchmark dataset is missing: {path}")
        try:
            records = self._read_records(path)
            if not isinstance(records, list):
                raise EvaluationError("Benchmark dataset must contain a list of records")
            return [DatasetRecord.model_validate(record) for record in records]
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise EvaluationError(f"Invalid benchmark dataset '{path}': {error}") from error

    @staticmethod
    def _read_records(path: Path) -> object:
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        elif suffix == ".csv":
            payload = pd.read_csv(path).fillna("").to_dict(orient="records")
        else:
            raise EvaluationError("Benchmark dataset must use .json, .yaml, .yml, or .csv")
        return payload.get("records", payload) if isinstance(payload, dict) else payload
