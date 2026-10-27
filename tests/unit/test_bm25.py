from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import Document, Entity, SearchFilters
from app.retrieval.bm25_memory import InMemoryBM25Retriever
from app.retrieval.elasticsearch_retriever import INDEX_MAPPINGS, ElasticsearchRetriever, build_query, to_es_document

DOCS = [
    Document(
        document_id="d1",
        title="MySQL deadlocks",
        content="InnoDB rolled back a transaction after a deadlock.",
        category="databases",
        source="a",
    ),
    Document(
        document_id="d2",
        title="Redis caching",
        content="Cache aside pattern reduces database reads.",
        category="databases",
        source="b",
    ),
    Document(
        document_id="d3",
        title="TCP handshake",
        content="TCP uses a three way handshake before data flows.",
        category="networking",
        source="a",
        entities=[Entity(text="TCP", label="ORG")],
    ),
]


@pytest.fixture
def bm25():
    r = InMemoryBM25Retriever()
    r.index_documents(DOCS)
    return r


def test_exact_term_match_ranks_first(bm25):
    results = bm25.retrieve("deadlock", top_k=3)
    assert results[0].document_id == "d1"
    assert results[0].retrieval_method == "bm25"
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_no_matching_terms_returns_empty(bm25):
    assert bm25.retrieve("kubernetes", top_k=5) == []
    assert bm25.retrieve("the and of", top_k=5) == []  # stopwords only


def test_title_match_outranks_body_match():
    r = InMemoryBM25Retriever()
    r.index_documents(
        [
            Document(document_id="title", title="Database sharding", content="Split rows across nodes for scale."),
            Document(document_id="body", title="Scaling notes", content="Sometimes sharding is used to split rows."),
        ]
    )
    assert r.retrieve("sharding", 2)[0].document_id == "title"


def test_matches_across_fields_accumulate(bm25):
    # "database" appears only in d2's body; "tcp" appears in d3's title, content and entities.
    scores = {r.document_id: r.score for r in bm25.retrieve("tcp database", 3)}
    assert scores["d3"] > scores["d2"]


def test_filters_and_removal(bm25):
    filtered = bm25.retrieve("tcp deadlock", 5, SearchFilters(category="networking"))
    assert [r.document_id for r in filtered] == ["d3"]
    bm25.remove_documents(["d3"])
    assert bm25.count() == 2
    assert all(r.document_id != "d3" for r in bm25.retrieve("tcp", 5))


def test_reindexing_same_document_replaces_it(bm25):
    bm25.index_documents([Document(document_id="d1", title="Something else", content="kubernetes pods")])
    assert bm25.count() == 3
    assert bm25.retrieve("deadlock", 3) == []
    assert bm25.retrieve("kubernetes", 3)[0].document_id == "d1"


# ------------------------------------------------------------------ Elasticsearch (mocked client)
def test_es_create_index_uses_mapping_when_missing():
    client = MagicMock()
    client.indices.exists.return_value = False
    ElasticsearchRetriever(client, "docs").create_index()
    kwargs = client.indices.create.call_args.kwargs
    assert kwargs["index"] == "docs"
    assert kwargs["mappings"] is INDEX_MAPPINGS
    assert kwargs["mappings"]["properties"]["content"]["type"] == "text"


def test_es_create_index_skips_existing_unless_recreate():
    client = MagicMock()
    client.indices.exists.return_value = True
    es = ElasticsearchRetriever(client, "docs")
    es.create_index()
    client.indices.create.assert_not_called()
    es.create_index(recreate=True)
    client.indices.delete.assert_called_once_with(index="docs")
    client.indices.create.assert_called_once()


def test_es_retrieve_maps_hits_to_results():
    client = MagicMock()
    client.search.return_value = {"hits": {"hits": [{"_id": "d3", "_score": 4.2}, {"_id": "d1", "_score": 1.5}]}}
    results = ElasticsearchRetriever(client, "docs").retrieve("tcp", 2, SearchFilters(category="networking"))
    assert [(r.document_id, r.score, r.rank) for r in results] == [("d3", 4.2, 1), ("d1", 1.5, 2)]
    query = client.search.call_args.kwargs["query"]
    assert {"term": {"category": "networking"}} in query["bool"]["filter"]


def test_es_query_targets_title_content_entities():
    body = build_query("aws security", 7, None)
    assert body["size"] == 7
    match = body["query"]["bool"]["must"][0]["multi_match"]
    assert match["fields"] == ["title^2", "content", "entities.text"]
    assert body["query"]["bool"]["filter"] == []


def test_es_bulk_index_builds_actions():
    client = MagicMock()
    with patch("elasticsearch.helpers.bulk", return_value=(3, [])) as bulk:
        assert ElasticsearchRetriever(client, "docs").index_documents(DOCS) == 3
    actions = list(bulk.call_args.args[1])
    assert [a["_id"] for a in actions] == ["d1", "d2", "d3"]
    assert actions[2]["_source"]["entities"] == [{"text": "TCP", "label": "ORG"}]


def test_es_is_available_false_on_connection_error():
    client = MagicMock()
    client.ping.side_effect = ConnectionError("down")
    assert ElasticsearchRetriever(client, "docs").is_available() is False


def test_to_es_document_serialises_fields():
    payload = to_es_document(DOCS[0])
    assert payload["document_id"] == "d1" and payload["category"] == "databases"
    assert set(payload) <= set(INDEX_MAPPINGS["properties"])
