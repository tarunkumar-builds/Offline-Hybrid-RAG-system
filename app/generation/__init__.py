"""Reserved for a future local generation layer."""
"""Offline Ollama-backed grounded answer generation."""

from app.generation.answer_generator import AnswerGenerator
from app.generation.config import GenerationConfig
from app.generation.models import Citation, GeneratedAnswer
from app.generation.pipeline import GenerationPipeline

__all__ = ["AnswerGenerator", "Citation", "GeneratedAnswer", "GenerationConfig", "GenerationPipeline"]
