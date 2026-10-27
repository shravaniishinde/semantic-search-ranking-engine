"""Common retrieval interface."""

from __future__ import annotations

from typing import Protocol

from app.models.schemas import Document, RetrievalResult, SearchFilters


class Retriever(Protocol):
    method: str

    def retrieve(self, query: str, top_k: int, filters: SearchFilters | None = None) -> list[RetrievalResult]: ...


class LexicalRetriever(Retriever, Protocol):
    def index_documents(self, documents: list[Document]) -> int: ...

    def remove_documents(self, document_ids: list[str]) -> None: ...

    def count(self) -> int: ...


def passes_filters(doc: Document, filters: SearchFilters | None) -> bool:
    if filters is None:
        return True
    if filters.category and doc.category.lower() != filters.category.lower():
        return False
    if filters.source and doc.source.lower() != filters.source.lower():
        return False
    return True
