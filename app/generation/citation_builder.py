"""Build source citations from labelled prompt context."""

import re
from collections.abc import Sequence

from app.generation.models import Citation
from app.reranker.models import RerankedChunk


class CitationBuilder:
    """Convert cited source labels into stable document metadata citations."""

    _source_pattern = re.compile(r"\[S(\d+)\]")

    def build(self, answer: str, chunks: Sequence[RerankedChunk]) -> list[Citation]:
        """Return citations mentioned by the answer, or all context as a safe fallback."""
        referenced = {int(match) for match in self._source_pattern.findall(answer)}
        selected = [
            chunk for index, chunk in enumerate(chunks, start=1) if index in referenced
        ] or list(chunks)
        return [
            Citation(
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                chunk_number=chunk.chunk_number,
                snippet=self._snippet(chunk.text),
            )
            for chunk in selected
        ]

    @staticmethod
    def _snippet(text: str, limit: int = 240) -> str:
        normalized = " ".join(text.split())
        return normalized if len(normalized) <= limit else f"{normalized[: limit - 3]}..."
