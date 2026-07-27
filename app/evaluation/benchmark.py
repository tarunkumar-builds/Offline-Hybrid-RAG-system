"""Dataset-scale evaluation execution and summary aggregation."""

from collections.abc import Callable, Sequence

from loguru import logger

from app.evaluation.evaluator import Evaluator
from app.evaluation.models import BenchmarkResult, BenchmarkSummary, DatasetRecord, EvaluationInput, EvaluationResult
from app.utils.errors import EvaluationError


class BenchmarkRunner:
    """Run injected RAG execution against a local dataset and summarize results."""

    def __init__(self, evaluator: Evaluator | None = None) -> None:
        self._evaluator = evaluator or Evaluator()

    def run(
        self,
        records: Sequence[DatasetRecord],
        execute: Callable[[DatasetRecord], EvaluationInput],
    ) -> BenchmarkResult:
        """Evaluate every executable record while retaining successful results."""
        results: list[EvaluationResult] = []
        for index, record in enumerate(records, start=1):
            try:
                logger.info("Benchmark progress: {}/{}", index, len(records))
                evaluation_input = execute(record)
                evaluation_input = evaluation_input.model_copy(
                    update={
                        "reference_answer": evaluation_input.reference_answer or record.reference_answer,
                        "expected_documents": evaluation_input.expected_documents or record.expected_documents,
                    }
                )
                result = self._evaluator.evaluate(evaluation_input)
                results.append(result)
            except Exception as error:
                logger.exception("Benchmark evaluation failed at record {}", index)
                raise EvaluationError("Benchmark execution failed") from error
        return BenchmarkResult(results=tuple(results), summary=self._summarize(records, results))

    @staticmethod
    def _summarize(records: Sequence[DatasetRecord], results: Sequence[EvaluationResult]) -> BenchmarkSummary:
        successful = [result for result in results if result.generation.answer_available]
        ranked = sorted(results, key=lambda item: (item.citations.citation_coverage, item.generation.f1 or 0.0))
        return BenchmarkSummary(
            total_queries=len(records),
            successful_queries=len(successful),
            success_rate=len(successful) / len(records) if records else 0.0,
            average_latency=(
                sum(result.performance.total_pipeline_time for result in results) / len(results) if results else 0.0
            ),
            average_citation_coverage=(
                sum(result.citations.citation_coverage for result in results) / len(results) if results else 0.0
            ),
            best_question=ranked[-1].question if ranked else None,
            worst_question=ranked[0].question if ranked else None,
        )
