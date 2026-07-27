"""Metadata filter helpers for retrieval results."""

from collections.abc import Iterable
from typing import TypeVar

from app.retrieval.models import SearchFilters


Filterable = TypeVar("Filterable", bound=object)


def matches_filters(item: Filterable, filters: SearchFilters | None) -> bool:
    """Return whether a metadata-bearing item satisfies all requested filters."""
    if filters is None:
        return True
    return (
        (filters.document_name is None or getattr(item, "document_name") == filters.document_name)
        and (filters.document_id is None or getattr(item, "document_id") == filters.document_id)
        and (filters.page_number is None or getattr(item, "page_number") == filters.page_number)
    )


def apply_filters(items: Iterable[Filterable], filters: SearchFilters | None) -> list[Filterable]:
    """Materialize only the items that match optional metadata constraints."""
    return [item for item in items if matches_filters(item, filters)]
