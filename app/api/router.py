"""Versioned API router composition."""

from fastapi import APIRouter

from app.api.routes import (
    documents,
    evaluation,
    health,
    query,
    system,
)

api_router = APIRouter(
    prefix="/api/v1",
    tags=["API v1"],
)

# ---------------------------------------------------------
# Document Management
# ---------------------------------------------------------

api_router.include_router(documents.router)

# ---------------------------------------------------------
# Question Answering
# ---------------------------------------------------------

api_router.include_router(query.router)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

api_router.include_router(evaluation.router)

# ---------------------------------------------------------
# Health Monitoring
# ---------------------------------------------------------

api_router.include_router(health.router)

# ---------------------------------------------------------
# System Information
# ---------------------------------------------------------

api_router.include_router(system.router)