"""Runs only when a real Elasticsearch is reachable (``docker compose up elasticsearch``)."""

import os
import uuid

import pytest

from app.models.schemas import Document, SearchFilters
from app.retrieval.elasticsearch_retriever import ElasticsearchRetriever

pytestmark = [pytest.mark.integration, pytest.mark.elasticsearch]

ES_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")


@pytest.fixture
def es():
    retriever = ElasticsearchRetriever.from_url(ES_URL, f"test-{uuid.uuid4().hex[:8]}", timeout=2.0)
    if not retriever.is_available():
        # CI's Elasticsearch job sets REQUIRE_ELASTICSEARCH=1 so a broken cluster fails loudly.
        if os.environ.get("REQUIRE_ELASTICSEARCH") == "1":
            pytest.fail(f"Elasticsearch required but not reachable at {ES_URL}")
        pytest.skip(f"Elasticsearch not reachable at {ES_URL}")
    retriever.create_index(recreate=True)
    yield retriever
    retriever.client.indices.delete(index=retriever.index_name, ignore_unavailable=True)


def test_bulk_index_and_bm25_search(es):
    docs = [
        Document(
            document_id="a",
            title="MySQL deadlocks",
            content="InnoDB detected a deadlock between transactions.",
            category="databases",
        ),
        Document(
            document_id="b", title="TCP handshake", content="Three-way handshake establishes connections.", category="networking"
        ),
    ]
    assert es.index_documents(docs) == 2
    assert es.count() == 2
    results = es.retrieve("deadlocks", 5)
    assert results[0].document_id == "a" and results[0].score > 0
    assert [r.document_id for r in es.retrieve("handshake deadlock", 5, SearchFilters(category="networking"))] == ["b"]
    es.remove_documents(["a"])
    es.client.indices.refresh(index=es.index_name)
    assert es.count() == 1
