"""Answer creation service that combines prompts, Ollama, and citations."""

from collections.abc import Sequence
from time import perf_counter

from loguru import logger

from app.generation.citation_builder import CitationBuilder
from app.generation.config import GenerationConfig
from app.generation.models import GeneratedAnswer
from app.generation.ollama_client import OllamaClient
from app.generation.prompt_builder import PromptBuilder
from app.reranker.models import RerankedChunk
from app.utils.errors import GenerationError


class AnswerGenerator:
    """Generate a grounded structured answer from reranked local context."""

    def __init__(
        self,
        config: GenerationConfig,
        prompt_builder: PromptBuilder | None = None,
        ollama_client: OllamaClient | None = None,
        citation_builder: CitationBuilder | None = None,
    ) -> None:
        self._config = config
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._ollama_client = ollama_client or OllamaClient(config)
        self._citation_builder = citation_builder or CitationBuilder()

    def generate(self, question: str, context_chunks: Sequence[RerankedChunk]) -> GeneratedAnswer:
        """Build a grounded prompt, call Ollama, and attach source citations."""
        if not context_chunks:
            raise GenerationError("Cannot generate an answer without reranked context")
        logger.bind(event="generation_started").info("Generating grounded answer from {} context chunks", len(context_chunks))
        prompt = self._prompt_builder.build(question, context_chunks, self._config.prompt_template)
        prompt_tokens = self._prompt_builder.estimate_tokens(prompt)
        logger.bind(event="prompt_created").info("Prompt created with approximately {} tokens", prompt_tokens)
        start = perf_counter()
        answer = self._ollama_client.generate(prompt)
        generation_time = perf_counter() - start
        if not answer.strip():
            raise GenerationError("Ollama returned an empty answer")
        citations = self._citation_builder.build(answer, context_chunks)
        logger.bind(event="generation_completed").info("Model response completed in {:.3f}s", generation_time)
        return GeneratedAnswer(
            answer=answer,
            citations=tuple(citations),
            source_documents=tuple(dict.fromkeys(chunk.document_name for chunk in context_chunks)),
            retrieved_chunks=tuple(context_chunks),
            model_name=self._config.model_name,
            generation_time=generation_time,
            prompt_tokens=prompt_tokens,
            response_tokens=self._prompt_builder.estimate_tokens(answer),
        )

    def close(self) -> None:
        """Release the persistent Ollama connection owned by this generator."""
        self._ollama_client.close()
