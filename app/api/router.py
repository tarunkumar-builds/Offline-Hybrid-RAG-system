"""Versioned router composition."""

from fastapi import APIRouter

from app.api.routes import documents, evaluation, health, query, system


api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents.router)
api_router.include_router(query.router)
api_router.include_router(evaluation.router)
api_router.include_router(health.router)
api_router.include_router(system.router)
