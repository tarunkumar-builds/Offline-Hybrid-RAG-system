"""Request logging and CORS middleware configuration."""

from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger


def configure_middleware(app: FastAPI) -> None:
    """Attach local-development CORS and structured request timing logs."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        start = perf_counter()
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request_logger = logger.bind(event="http_request", request_id=request_id)
        try:
            response = await call_next(request)
        except Exception:
            request_logger.exception("Unhandled request failure")
            raise
        duration = perf_counter() - start
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        response.headers["X-Request-ID"] = request_id
        request_logger.info("{} {} -> {} in {:.3f}s", request.method, request.url.path, response.status_code, duration)
        return response
