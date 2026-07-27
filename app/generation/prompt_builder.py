"""Grounded prompt construction from reranked document chunks."""

from collections.abc import Sequence

from app.generation.prompt_loader import PromptLoader
from app.reranker.models import RerankedChunk
from app.utils.errors import GenerationError


class PromptBuilder:
    """Build consistent, citation-aware prompts from named YAML templates."""

    def __init__(self, prompt_loader: PromptLoader | None = None) -> None:
        self._prompt_loader = prompt_loader or PromptLoader()

    def build(self, question: str, chunks: Sequence[RerankedChunk], template_name: str) -> str:
        """Create a prompt containing instructions, labelled context, and question."""
        if not question.strip():
            raise GenerationError("Question must not be blank")
        if not chunks:
            raise GenerationError("Cannot generate an answer without retrieved context")
        template = self._prompt_loader.load(template_name)
        context = "\n\n".join(
            self._format_chunk(index, chunk) for index, chunk in enumerate(chunks, start=1)
        )
        return (
            "SYSTEM INSTRUCTIONS:\n"
            f"{template['system_instructions'].strip()}\n\n"
            "RESPONSE RULES:\n"
            f"{template['response_rules'].strip()}\n\n"
            "CITATION INSTRUCTIONS:\n"
            f"{template['citation_instructions'].strip()}\n\n"
            "RETRIEVED CONTEXT:\n"
            f"{context}\n\n"
            "USER QUESTION:\n"
            f"{question.strip()}\n\n"
            "ANSWER:"
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Provide a lightweight whitespace-based token estimate for observability."""
        return len(text.split())

    @staticmethod
    def _format_chunk(index: int, chunk: RerankedChunk) -> str:
        return (
            f"[S{index}] Document: {chunk.document_name}\n"
            f"Page: {chunk.page_number}\n"
            f"Chunk: {chunk.chunk_number}\n"
            f"Text: {chunk.text}"
        )
