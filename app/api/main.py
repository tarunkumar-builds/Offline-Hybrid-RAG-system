"""FastAPI application factory for the offline Hybrid RAG backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.api.dependencies import get_services, get_settings
from app.api.exception_handlers import configure_exception_handlers
from app.api.middleware import configure_middleware
from app.api.router import api_router
from app.utils.logging import configure_logging


def create_app() -> FastAPI:
    """Create the documented versioned REST API without eager model loading."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        services = get_services()
        logger.bind(event="application_started").info("Offline Hybrid RAG API started")
        try:
            yield
        finally:
            services.close()
            get_services.cache_clear()
            logger.bind(event="application_stopped").info("Offline Hybrid RAG API stopped")

    app = FastAPI(
        title="Offline Hybrid RAG API",
        version="1.0.0",
        description="Local PDF ingestion, hybrid retrieval, reranking, Ollama generation, and evaluation.",
        openapi_tags=[
            {"name": "Documents", "description": "Local PDF ingestion and lifecycle."},
            {"name": "Query", "description": "Grounded local question answering."},
            {"name": "Evaluation", "description": "Offline single-query and benchmark metrics."},
            {"name": "Health", "description": "Backend and dependency status."},
            {"name": "System", "description": "Read-only configuration and model details."},
        ],
        lifespan=lifespan,
    )
    configure_logging(settings.log_level)
    configure_middleware(app)
    configure_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
