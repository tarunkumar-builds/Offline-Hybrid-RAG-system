"""Consistent JSON responses for validation and domain failures."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from app.utils.errors import (
    EvaluationError,
    GenerationError,
    IngestionError,
    OllamaConnectionError,
    OllamaResponseError,
    RetrievalConfigurationError,
    RetrievalError,
    RerankingError,
)


def configure_exception_handlers(app: FastAPI) -> None:
    """Register API-safe handlers without exposing internal stack details."""
    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        details = [
            {"type": item["type"], "loc": item["loc"], "msg": item["msg"]}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    async def domain_error(_: Request, error: Exception) -> JSONResponse:
        status_code, detail = _domain_response(error)
        logger.bind(event="api_domain_error", error_type=type(error).__name__).warning("API domain error")
        return JSONResponse(status_code=status_code, content={"detail": detail})

    for error_type in (IngestionError, RetrievalError, RerankingError, GenerationError, EvaluationError):
        app.add_exception_handler(error_type, domain_error)

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, error: Exception) -> JSONResponse:
        logger.bind(event="api_unexpected_error", error_type=type(error).__name__).exception("Unhandled API error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _domain_response(error: Exception) -> tuple[int, str]:
    if isinstance(error, RetrievalConfigurationError):
        return 409, "No indexed documents are available for retrieval"
    if isinstance(error, OllamaConnectionError):
        return 503, "Local generation service is unavailable"
    if isinstance(error, OllamaResponseError):
        return 502, "Local generation service rejected the request"
    if isinstance(error, (RetrievalError, RerankingError, GenerationError)):
        return 503, "The retrieval and generation service is temporarily unavailable"
    if isinstance(error, EvaluationError):
        return 400, "Evaluation could not be completed"
    if isinstance(error, IngestionError):
        return 400, "Document could not be ingested"
    return 500, "Internal server error"
