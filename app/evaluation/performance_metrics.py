"""Normalize optional stage timing data into a stable metric contract."""

from app.evaluation.models import PerformanceMetrics
from app.generation.models import GeneratedAnswer


def calculate_performance_metrics(
    timings: dict[str, float], answer: GeneratedAnswer | None
) -> PerformanceMetrics:
    """Return known stage metrics, defaulting unavailable instrumentation to zero."""
    values = {key: max(0.0, float(value)) for key, value in timings.items()}
    values.setdefault("llm_generation_time", answer.generation_time if answer else 0.0)
    if "total_pipeline_time" not in values:
        values["total_pipeline_time"] = sum(
            value for key, value in values.items() if key != "total_pipeline_time"
        )
    return PerformanceMetrics(**{key: values.get(key, 0.0) for key in PerformanceMetrics.model_fields})
