"""Text normalization that keeps paragraph boundaries intact."""

import re
import unicodedata


class TextCleaner:
    """Normalize extracted PDF text while preserving semantic paragraphs."""

    _paragraph_break = re.compile(r"\n\s*\n+")
    _horizontal_whitespace = re.compile(r"[\t \f\v]+")
    _single_newline = re.compile(r"\s*\n\s*")

    def clean(self, text: str) -> str:
        """Remove control noise, collapse broken lines, and retain paragraphs."""
        normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
        printable = "".join(character for character in normalized if character.isprintable() or character == "\n")
        paragraphs = self._paragraph_break.split(printable)
        cleaned = []
        for paragraph in paragraphs:
            paragraph = self._single_newline.sub(" ", paragraph)
            paragraph = self._horizontal_whitespace.sub(" ", paragraph).strip()
            if paragraph:
                cleaned.append(paragraph)
        return "\n\n".join(cleaned)
