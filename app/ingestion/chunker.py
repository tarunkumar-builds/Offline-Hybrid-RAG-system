"""Sentence-aware chunking with configurable token overlap."""

import re
from collections.abc import Callable
from uuid import NAMESPACE_URL, uuid5

from app.models import Chunk, PageText


class TokenChunker:
    """Build page-scoped, sentence-aware chunks under a token budget."""

    _sentence_boundary = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
    _word_tokens = re.compile(r"\S+")

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_pages(
        self,
        pages: list[PageText],
        document_id: str,
        count_tokens: Callable[[str], int] | None = None,
    ) -> list[Chunk]:
        """Split cleaned pages into deterministic chunks in document order."""
        token_counter = count_tokens or self._fallback_token_count
        chunks: list[Chunk] = []
        chunk_number = 0
        for page in pages:
            for text in self._chunk_text(page.text, token_counter):
                identifier = str(uuid5(NAMESPACE_URL, f"{document_id}:{page.page_number}:{chunk_number}:{text}"))
                chunks.append(
                    Chunk(
                        chunk_id=identifier,
                        document_id=document_id,
                        page_number=page.page_number,
                        chunk_number=chunk_number,
                        text=text,
                    )
                )
                chunk_number += 1
        return chunks

    def _chunk_text(self, text: str, count_tokens: Callable[[str], int]) -> list[str]:
        units = self._sentence_units(text)
        chunks: list[str] = []
        current: list[str] = []
        for unit in units:
            if current and count_tokens(" ".join(current + [unit])) > self._chunk_size:
                chunks.append(" ".join(current))
                current = self._overlap_units(current, count_tokens)
            if count_tokens(unit) > self._chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                chunks.extend(self._split_long_unit(unit, count_tokens))
            else:
                current.append(unit)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _sentence_units(self, text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        return [sentence.strip() for paragraph in paragraphs for sentence in self._sentence_boundary.split(paragraph) if sentence.strip()]

    def _overlap_units(self, units: list[str], count_tokens: Callable[[str], int]) -> list[str]:
        overlap: list[str] = []
        for unit in reversed(units):
            candidate = [unit, *overlap]
            if count_tokens(" ".join(candidate)) > self._chunk_overlap:
                break
            overlap = candidate
        return overlap

    def _split_long_unit(self, unit: str, count_tokens: Callable[[str], int]) -> list[str]:
        words = self._word_tokens.findall(unit)
        segments: list[str] = []
        current: list[str] = []
        for word in words:
            if current and count_tokens(" ".join(current + [word])) > self._chunk_size:
                segments.append(" ".join(current))
                current = self._overlap_units(current, count_tokens)
            current.append(word)
        if current:
            segments.append(" ".join(current))
        return segments

    def _fallback_token_count(self, text: str) -> int:
        return len(self._word_tokens.findall(text))
