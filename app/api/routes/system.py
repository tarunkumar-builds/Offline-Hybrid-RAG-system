"""Read-only configuration and model endpoints."""

from fastapi import APIRouter, Depends

from app.api.dependencies import ApiServices, get_services
from app.api.schemas.health import ModelsResponse, SystemConfigResponse


router = APIRouter(prefix="/system", tags=["System"])


@router.get("/config", response_model=SystemConfigResponse)
def system_config(services: ApiServices = Depends(get_services)) -> SystemConfigResponse:
    """Return safe runtime controls without filesystem or connection details."""
    return SystemConfigResponse(configuration=services.public_config())


@router.get("/models", response_model=ModelsResponse)
def system_models(services: ApiServices = Depends(get_services)) -> ModelsResponse:
    return ModelsResponse(
        embedding_model=services.settings.embedding_model,
        reranker_model=services.settings.reranker_model,
        llm_model=services.settings.ollama_model,
    )
