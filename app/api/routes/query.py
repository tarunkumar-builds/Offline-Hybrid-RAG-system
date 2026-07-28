"""Question-answering API endpoint."""

import time

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.api.dependencies import ApiServices, get_services
from app.api.schemas.query import QueryRequest, QueryResponse
from app.evaluation.models import EvaluationInput


router = APIRouter(
    prefix="/query",
    tags=["Query"],
)


@router.post("", response_model=QueryResponse)
def answer_query(
    request: QueryRequest,
    services: ApiServices = Depends(get_services),
) -> QueryResponse:
    """
    Execute the complete Offline Hybrid RAG pipeline.

    Pipeline:

    User Question
        ↓
    Hybrid Retrieval
        ↓
    Cross Encoder Reranking
        ↓
    Prompt Builder
        ↓
    Ollama Generation
        ↓
    Optional Evaluation
        ↓
    Structured JSON Response
    """

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    logger.info(f"Incoming query: {request.question}")

    start_time = time.perf_counter()

    try:
        answer = services.answer(
            request.question,
            request.top_k,
            request.model_name,
            request.temperature,
            request.prompt_template,
        )

        if not answer.retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found.",
            )

        evaluation = None

        if request.evaluation_enabled:
            result = services.evaluation.evaluate(
                EvaluationInput(
                    question=request.question,
                    reranked_chunks=answer.retrieved_chunks,
                    generated_answer=answer,
                )
            )

            evaluation = result.model_dump(mode="json")

        processing_time = round(
            time.perf_counter() - start_time,
            3,
        )

        logger.info(
            f"Query completed successfully in "
            f"{processing_time}s "
            f"using model {answer.model_name}"
        )

        return QueryResponse(
            answer=answer.answer,
            citations=answer.citations,
            retrieved_chunks=answer.retrieved_chunks,
            model_name=answer.model_name,
            generation_time=answer.generation_time,
            processing_time=processing_time,
            evaluation=evaluation,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Query pipeline failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(exc)}",
        ) from exc