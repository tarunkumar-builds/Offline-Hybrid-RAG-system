"""High-level single-query evaluation and reporting API."""

from app.config.settings import Settings
from app.evaluation.config import EvaluationConfig
from app.evaluation.evaluator import Evaluator
from app.evaluation.models import EvaluationInput, EvaluationResult
from app.evaluation.report_generator import ReportGenerator


class EvaluationPipeline:
    """Evaluate supplied RAG artifacts and optionally persist one report."""

    def __init__(self, evaluator: Evaluator, report_generator: ReportGenerator, config: EvaluationConfig) -> None:
        self._evaluator = evaluator
        self._report_generator = report_generator
        self._config = config

    @classmethod
    def from_settings(cls, settings: Settings) -> "EvaluationPipeline":
        """Build an evaluation pipeline from environment-aware application settings."""
        config = EvaluationConfig.from_settings(settings)
        return cls(Evaluator(), ReportGenerator(config.output_directory), config)

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        """Evaluate one query without writing a report."""
        return self._evaluator.evaluate(evaluation_input)

    def evaluate_and_report(self, evaluation_input: EvaluationInput, name: str = "evaluation") -> EvaluationResult:
        """Evaluate one query and write it using the configured report format."""
        result = self.evaluate(evaluation_input)
        self._report_generator.write_evaluation(result, self._config.report_format, name)
        return result
