"""Health and system status API contracts."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    embedding_model: str
    reranker_model: str
    indexed_documents: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    database_status: str


class SystemConfigResponse(BaseModel):
    configuration: dict[str, object]


class ModelsResponse(BaseModel):
    embedding_model: str
    reranker_model: str
    llm_model: str
