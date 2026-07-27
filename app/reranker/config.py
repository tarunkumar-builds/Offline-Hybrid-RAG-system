"""Configuration object for the local cross-encoder reranker."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config.settings import Settings


class RerankerConfig(BaseModel):
    """Runtime parameters controlling model loading and result limits."""

    model_config = ConfigDict(frozen=True)

    model_name: str = "BAAI/bge-reranker-base"
    top_k_input: int = Field(default=20, ge=1)
    top_n_output: int = Field(default=5, ge=1)
    batch_size: int = Field(default=16, ge=1)
    device: str = "cpu"
    cache_size: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def validate_limits(self) -> "RerankerConfig":
        """Ensure the requested output is no larger than its input pool."""
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        if self.top_n_output > self.top_k_input:
            raise ValueError("top_n_output cannot exceed top_k_input")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> "RerankerConfig":
        """Create a reranker configuration from application settings."""
        return cls(
            model_name=settings.reranker_model,
            top_k_input=settings.reranker_top_k_input,
            top_n_output=settings.reranker_top_n_output,
            batch_size=settings.reranker_batch_size,
            device=settings.reranker_device,
            cache_size=settings.reranker_cache_size,
        )
