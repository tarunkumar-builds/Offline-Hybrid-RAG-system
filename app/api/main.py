"""FastAPI application factory for the Offline Hybrid RAG backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.exception_handlers import configure_exception_handlers
from app.api.middleware import configure_middleware
from app.api.router import api_router
from app.utils.logging import configure_logging
from loguru import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize application-wide singleton resources.

    Heavy resources should be loaded only once during startup and
    cleaned up during shutdown.
    """

    logger.info("Initializing Offline Hybrid RAG services...")

    # ---------------------------------------------------------
    # TODO: Replace these placeholders with your actual services
    # ---------------------------------------------------------

    app.state.embedding_model = None
    app.state.reranker = None
    app.state.ollama_client = None
    app.state.vector_store = None
    app.state.database = None
    app.state.settings = None

    logger.info("Application startup complete.")

    yield

    logger.info("Shutting down Offline Hybrid RAG services...")

    # Close database connections, clients, etc. here if needed.


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    configure_logging("INFO")

    app = FastAPI(
        title="Offline Hybrid RAG API",
        version="1.0.0",
        description=(
            "Production-ready offline Hybrid RAG backend supporting "
            "document ingestion, hybrid retrieval, reranking, "
            "Ollama generation, and evaluation."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {
                "name": "Documents",
                "description": "PDF upload, indexing and lifecycle management.",
            },
            {
                "name": "Query",
                "description": "Grounded question answering.",
            },
            {
                "name": "Evaluation",
                "description": "Offline evaluation and benchmarking.",
            },
            {
                "name": "Health",
                "description": "Application health and dependency status.",
            },
            {
                "name": "System",
                "description": "Application configuration and model information.",
            },
        ],
    )

    configure_middleware(app)
    configure_exception_handlers(app)

    app.include_router(api_router)

    @app.get("/", tags=["System"])
    async def root():
        """Root endpoint."""

        return {
            "application": "Offline Hybrid RAG",
            "version": "1.0.0",
            "status": "running",
            "documentation": "/docs",
            "health": "/api/v1/health",
        }

    logger.info("FastAPI application created successfully.")

    return app


app = create_app()