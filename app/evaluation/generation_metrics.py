"""Offline answer availability and lexical reference metric calculations."""

import math
import re

from app.evaluation.models import GenerationMetrics
from app.generation.models import GeneratedAnswer


_token_pattern = re.compile(r"[^\W_]+", re.UNICODE)


def calculate_generation_metrics(
    answer: GeneratedAnswer | None, reference_answer: str | None, has_citations: bool
) -> GenerationMetrics:
    """Compute answer metadata and optional local lexical similarity metrics."""
    text = answer.answer.strip() if answer else ""
    metrics = _reference_metrics(text, reference_answer) if reference_answer is not None else {}
    return GenerationMetrics(
        answer_length=len(text),
        generation_time=answer.generation_time if answer else 0.0,
        prompt_tokens=answer.prompt_tokens if answer else 0,
        completion_tokens=answer.response_tokens if answer else 0,
        answer_available=bool(text),
        empty_answer=not bool(text),
        unsupported_answer=bool(text) and not has_citations,
        **metrics,
    )


def _reference_metrics(answer: str, reference: str) -> dict[str, float]:
    answer_tokens = _tokens(answer)
    reference_tokens = _tokens(reference)
    answer_set, reference_set = set(answer_tokens), set(reference_tokens)
    overlap = len(answer_set & reference_set)
    precision = overlap / len(answer_set) if answer_set else 0.0
    recall = overlap / len(reference_set) if reference_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lcs = _lcs_length(answer_tokens, reference_tokens)
    rouge_l = 2 * lcs / (len(answer_tokens) + len(reference_tokens)) if answer_tokens or reference_tokens else 0.0
    bleu_precision = sum(token in reference_set for token in answer_tokens) / len(answer_tokens) if answer_tokens else 0.0
    brevity = min(1.0, math.exp(1 - len(reference_tokens) / len(answer_tokens))) if answer_tokens else 0.0
    return {
        "exact_match": float(_normalize(answer) == _normalize(reference)),
        "rouge_l": rouge_l,
        "bleu": bleu_precision * brevity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _tokens(text: str) -> list[str]:
    return _token_pattern.findall(text.lower())


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _lcs_length(first: list[str], second: list[str]) -> int:
    previous = [0] * (len(second) + 1)
    for first_token in first:
        current = [0]
        for index, second_token in enumerate(second, start=1):
            current.append(previous[index - 1] + 1 if first_token == second_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]
