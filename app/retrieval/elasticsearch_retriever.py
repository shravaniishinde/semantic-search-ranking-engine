"""Elasticsearch BM25 retriever.

The client is injected, so tests can pass a ``MagicMock`` and no cluster is needed.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.schemas import Document, RetrievalResult, SearchFilters

logger = logging.getLogger(__name__)

INDEX_SETTINGS: dict[str, Any] = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    # BM25 with Lucene defaults, stated explicitly for clarity.
    "similarity": {"default": {"type": "BM25", "k1": 1.2, "b": 0.75}},
}

INDEX_MAPPINGS: dict[str, Any] = {
    "dynamic": "strict",
    "properties": {
        "document_id": {"type": "keyword"},
        "title": {"type": "text", "analyzer": "english", "fields": {"raw": {"type": "keyword"}}},
        "content": {"type": "text", "analyzer": "english"},
        "category": {"type": "keyword"},
        "source": {"type": "keyword"},
        "created_at": {"type": "date"},
        "entities": {
            "properties": {
                "text": {"type": "text", "analyzer": "standard", "fields": {"raw": {"type": "keyword"}}},
                "label": {"type": "keyword"},
            }
        },
        "cluster_id": {"type": "integer"},
        "is_outlier": {"type": "boolean"},
        "metadata": {"type": "flattened"},
    },
}

SEARCH_FIELDS = ["title^2", "content", "entities.text"]


def to_es_document(doc: Document) -> dict[str, Any]:
    return {
        "document_id": doc.document_id,
        "title": doc.title,
        "content": doc.content,
        "category": doc.category,
        "source": doc.source,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "entities": [{"text": e.text, "label": e.label} for e in doc.entities],
        "cluster_id": doc.cluster_id,
        "is_outlier": doc.is_outlier,
        "metadata": {k: str(v) for k, v in doc.metadata.items()},
    }


def build_query(query: str, top_k: int, filters: SearchFilters | None) -> dict[str, Any]:
    must = {"multi_match": {"query": query, "fields": SEARCH_FIELDS, "type": "most_fields"}}
    filter_clauses = []
    if filters and filters.category:
        filter_clauses.append({"term": {"category": filters.category}})
    if filters and filters.source:
        filter_clauses.append({"term": {"source": filters.source}})
    return {"size": top_k, "query": {"bool": {"must": [must], "filter": filter_clauses}}}


class ElasticsearchRetriever:
    method = "bm25"
    backend = "elasticsearch"

    def __init__(self, client: Any, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    @classmethod
    def from_url(cls, url: str, index_name: str, timeout: float = 5.0) -> ElasticsearchRetriever:
        from elasticsearch import Elasticsearch

        # No transport-level retries: an unreachable cluster should be detected quickly so the
        # engine can fall back to in-memory BM25 instead of stalling startup.
        return cls(Elasticsearch(url, request_timeout=timeout, max_retries=0, retry_on_timeout=False), index_name)

    def is_available(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:  # noqa: BLE001 - any transport failure means "unavailable"
            return False

    def create_index(self, recreate: bool = False) -> None:
        exists = bool(self.client.indices.exists(index=self.index_name))
        if exists and recreate:
            self.client.indices.delete(index=self.index_name)
            exists = False
        if not exists:
            self.client.indices.create(index=self.index_name, settings=INDEX_SETTINGS, mappings=INDEX_MAPPINGS)
            logger.info("es.index_created", extra={"fields": {"index": self.index_name}})

    def index_document(self, doc: Document) -> None:
        self.client.index(index=self.index_name, id=doc.document_id, document=to_es_document(doc))

    def index_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        from elasticsearch import helpers

        actions = (
            {"_op_type": "index", "_index": self.index_name, "_id": d.document_id, "_source": to_es_document(d)}
            for d in documents
        )
        success, errors = helpers.bulk(self.client, actions, raise_on_error=False, refresh="wait_for")
        if errors:
            logger.error("es.bulk_errors", extra={"fields": {"count": len(errors), "first": str(errors[0])[:300]}})
        logger.info("es.bulk_indexed", extra={"fields": {"index": self.index_name, "count": success}})
        return int(success)

    def remove_documents(self, document_ids: list[str]) -> None:
        for doc_id in document_ids:
            self.client.options(ignore_status=404).delete(index=self.index_name, id=doc_id)

    def count(self) -> int:
        try:
            return int(self.client.count(index=self.index_name)["count"])
        except Exception:  # noqa: BLE001
            return 0

    def retrieve(self, query: str, top_k: int, filters: SearchFilters | None = None) -> list[RetrievalResult]:
        body = build_query(query, top_k, filters)
        response = self.client.search(index=self.index_name, query=body["query"], size=body["size"], source=False)
        hits = response["hits"]["hits"]
        return [
            RetrievalResult(document_id=h["_id"], score=round(float(h["_score"]), 6), rank=r, retrieval_method=self.method)
            for r, h in enumerate(hits, start=1)
        ]
