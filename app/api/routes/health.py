"""Local backend readiness endpoint."""

from fastapi import APIRouter, Depends

from app.api.dependencies import ApiServices, get_services
from app.api.schemas.health import HealthResponse


router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
def health(services: ApiServices = Depends(get_services)) -> HealthResponse:
    ollama, documents, vectors, database = services.health()
    return HealthResponse(
        status="healthy" if database == "ready" else "degraded",
        ollama_status=ollama,
        embedding_model=services.settings.embedding_model,
        reranker_model=services.settings.reranker_model,
        indexed_documents=documents,
        vector_count=vectors,
        database_status=database,
    )
