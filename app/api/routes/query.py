"""Question-answering API endpoint."""

from fastapi import APIRouter, Depends
from loguru import logger
from time import perf_counter

from app.api.dependencies import ApiServices, get_services
from app.api.schemas.query import QueryRequest, QueryResponse
from app.evaluation.models import EvaluationInput


router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
def answer_query(request: QueryRequest, services: ApiServices = Depends(get_services)) -> QueryResponse:
    """Run local retrieval, reranking, prompt creation, and Ollama generation."""
    start = perf_counter()
    logger.bind(event="query_received").info("Query request received")
    answer = services.answer(
        request.question, request.top_k, request.model_name, request.temperature, request.prompt_template
    )
    evaluation = None
    if request.evaluation_enabled:
        result = services.evaluation.evaluate(
            EvaluationInput(question=request.question, reranked_chunks=answer.retrieved_chunks, generated_answer=answer)
        )
        evaluation = result
    processing_time = perf_counter() - start
    logger.bind(event="query_completed").info("Query completed in {:.3f}s", processing_time)
    return QueryResponse(
        answer=answer.answer,
        citations=answer.citations,
        retrieved_chunks=answer.retrieved_chunks,
        model_name=answer.model_name,
        generation_time=answer.generation_time,
        processing_time=processing_time,
        evaluation=evaluation,
    )
