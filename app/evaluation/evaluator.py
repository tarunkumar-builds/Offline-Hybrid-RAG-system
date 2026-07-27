"""Coordinator for all single-query evaluation metrics."""

from loguru import logger

from app.evaluation.citation_metrics import calculate_citation_metrics
from app.evaluation.generation_metrics import calculate_generation_metrics
from app.evaluation.models import EvaluationInput, EvaluationResult
from app.evaluation.performance_metrics import calculate_performance_metrics
from app.evaluation.retrieval_metrics import calculate_retrieval_metrics


class Evaluator:
    """Evaluate completed retrieval, reranking, and generation artifacts offline."""

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        """Compute a stable evaluation result for one question."""
        logger.bind(event="evaluation_started").info("Evaluation started")
        answer = evaluation_input.generated_answer
        citations = evaluation_input.citations or (answer.citations if answer else ())
        answer_available = bool(answer and answer.answer.strip())
        result = EvaluationResult(
            question=evaluation_input.question,
            answer=answer.answer if answer else None,
            retrieved_documents=tuple(
                dict.fromkeys(
                    chunk.document_name
                    for chunk in (*evaluation_input.retrieved_chunks, *evaluation_input.reranked_chunks)
                )
            ),
            retrieval=calculate_retrieval_metrics(
                evaluation_input.retrieved_chunks, evaluation_input.reranked_chunks
            ),
            generation=calculate_generation_metrics(
                answer, evaluation_input.reference_answer, has_citations=bool(citations)
            ),
            citations=calculate_citation_metrics(citations, evaluation_input.reranked_chunks, answer_available),
            performance=calculate_performance_metrics(evaluation_input.timings, answer),
        )
        logger.bind(event="evaluation_completed").info("Evaluation metrics computed")
        return result
