"""Configuration for the local Ollama-backed generation engine."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config.settings import Settings


class GenerationConfig(BaseModel):
    """Generation settings supplied by the shared application configuration."""

    model_config = ConfigDict(frozen=True)

    model_name: str = "qwen3:latest"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout_seconds: float = Field(default=200.0, gt=0.0)
    prompt_template: str = "default"
    streaming: bool = False
    retries: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_values(self) -> "GenerationConfig":
        """Ensure names and local base URL are usable."""
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        if not self.base_url.strip():
            raise ValueError("base_url must not be blank")
        if not self.prompt_template.strip():
            raise ValueError("prompt_template must not be blank")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> "GenerationConfig":
        """Build a generation configuration from environment-aware settings."""
        return cls(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.generation_temperature,
            top_p=settings.generation_top_p,
            max_tokens=settings.generation_max_tokens,
            timeout_seconds=settings.generation_timeout_seconds,
            prompt_template=settings.generation_prompt_template,
            streaming=settings.generation_streaming,
            retries=settings.generation_retries,
        )
