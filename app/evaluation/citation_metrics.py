"""Citation completeness and duplication metric calculations."""

from collections.abc import Sequence

from app.evaluation.models import CitationMetrics
from app.generation.models import Citation
from app.reranker.models import RerankedChunk


def calculate_citation_metrics(
    citations: Sequence[Citation], reranked_chunks: Sequence[RerankedChunk], answer_available: bool
) -> CitationMetrics:
    """Measure source diversity, coverage of context chunks, and duplicates."""
    citation_keys = [(item.document_name, item.page_number, item.chunk_number) for item in citations]
    context_keys = {(item.document_name, item.page_number, item.chunk_number) for item in reranked_chunks}
    coverage = len(set(citation_keys) & context_keys) / len(context_keys) if context_keys else 0.0
    return CitationMetrics(
        citation_count=len(citations),
        unique_source_documents=len({item.document_name for item in citations}),
        unique_pages=len({(item.document_name, item.page_number) for item in citations}),
        citation_coverage=coverage,
        missing_citations=answer_available and not citations,
        duplicate_citations=len(citation_keys) - len(set(citation_keys)),
    )
