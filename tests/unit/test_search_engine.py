"""SearchEngine orchestration with lightweight components (hashing embedder, in-memory BM25)."""

import pytest

from app.embeddings.encoder import HashingEmbedder
from app.models.schemas import DocumentIn, SearchFilters
from app.retrieval.bm25_memory import InMemoryBM25Retriever
from app.services.search_service import QueryValidationError, SearchEngine
from tests.conftest import make_settings


def test_ingest_rejects_empty_documents_and_excludes_confirmed_duplicates(fast_engine):
    engine = fast_engine
    assert "misc-004" not in engine.documents  # whitespace-only content
    for dup, canonical in (("cloud-011", "cloud-006"), ("sec-011", "sec-001")):
        doc = engine.documents[dup]
        assert doc.is_duplicate and doc.duplicate_of == canonical
        assert not doc.indexed and dup not in engine.faiss_index
        assert engine.documents[canonical].indexed
    assert len(engine.faiss_index) == engine.lexical.count() == sum(d.indexed for d in engine.documents.values())


def test_ingested_documents_carry_entities_and_embedding_metadata(fast_engine):
    doc = fast_engine.documents["sec-001"]
    assert {"log4shell", "cve-2021-44228"} <= {e.text.lower() for e in doc.entities}
    assert doc.embedding.model_name == "hashing-bow-256" and doc.embedding.dimension == 256
    assert doc.cluster_id is not None


def test_html_noise_is_cleaned_on_ingest(fast_engine):
    content = fast_engine.documents["net-002"].content
    assert "<p>" not in content and "https://" not in content and "recursive resolver" in content


def test_search_returns_full_pipeline_output(fast_engine):
    r = fast_engine.search("Deadlock found when trying to get lock", top_k=5)
    assert r.results[0].document_id == "db-008"
    assert [i.rank for i in r.results] == [1, 2, 3, 4, 5]
    assert r.intent.label in {"troubleshooting", "entity_lookup", "technical", "informational", "comparison"}
    assert r.ranker == "xgboost_ltr"
    assert len(r.retrieval.bm25) <= 5 and r.retrieval.hybrid
    assert set(r.stage_latency_ms) == {
        "preprocessing",
        "ner",
        "intent",
        "query_embedding",
        "retrieval",
        "bm25",
        "semantic",
        "fusion",
        "features",
        "rerank",
    }
    # The retrieval breakdown is measured inside the "retrieval" stage, so it cannot exceed it.
    sub_stages = r.stage_latency_ms["bm25"] + r.stage_latency_ms["semantic"] + r.stage_latency_ms["fusion"]
    assert sub_stages <= r.stage_latency_ms["retrieval"] + 0.05
    # Counts describe the full candidate lists, not the top_k-truncated debug lists.
    assert r.retrieval_counts["hybrid"] == r.candidate_count >= len(r.results)
    assert r.retrieval_counts["semantic"] == 30
    top = r.results[0]
    assert top.features["in_bm25"] == 1.0 and top.bm25_rank == 1
    finals = [i.final_score for i in r.results]
    assert finals == sorted(finals, reverse=True)


def test_search_with_category_filter(fast_engine):
    r = fast_engine.search("security", top_k=10, filters=SearchFilters(category="networking"))
    assert r.results and all(i.category == "networking" for i in r.results)


def test_blank_query_rejected(fast_engine):
    with pytest.raises(QueryValidationError):
        fast_engine.search("  <b></b>  ")


def test_reingesting_same_id_updates_in_place(fast_engine):
    before = len(fast_engine.documents)
    report = fast_engine.ingest(
        [
            DocumentIn(
                document_id="se-002",
                title="Test-driven development",
                content="Write a failing test first, then make it pass with zebra crossing.",
            )
        ]
    )
    assert report.accepted == 1 and len(fast_engine.documents) == before
    assert fast_engine.search("zebra crossing", 3).results[0].document_id == "se-002"


def test_analytics_and_search_stats(fast_engine):
    fast_engine.search("tcp vs udp")
    a = fast_engine.analytics()
    assert a["document_count"] == len(fast_engine.documents)
    assert a["indexed_count"] + a["excluded_count"] == a["document_count"]
    assert a["duplicate_count"] >= 2
    assert a["search_stats"]["total_searches"] >= 1
    assert a["models"]["reranker"] == "xgboost_ltr"
    importance = a["models"]["reranker_feature_importance"]
    assert importance and abs(sum(importance.values()) - 1.0) < 0.01


def test_persistence_roundtrip(tmp_path, ner, sample_documents):
    settings = make_settings(tmp_path)
    engine = SearchEngine(settings, embedder=HashingEmbedder(), entity_extractor=ner, lexical=InMemoryBM25Retriever())
    engine.ingest(sample_documents)
    engine.ensure_models()
    expected = [r.document_id for r in engine.search("kubernetes pod restarting").results]

    restored = SearchEngine(settings, embedder=HashingEmbedder(), entity_extractor=ner, lexical=InMemoryBM25Retriever())
    assert restored.load_state()
    restored.ensure_models(train_if_missing=False)  # must load from disk, not retrain
    assert restored.reranker.is_trained and restored.intent.is_trained
    assert [r.document_id for r in restored.search("kubernetes pod restarting").results] == expected


def test_list_documents_filters(fast_engine):
    total, docs = fast_engine.list_documents(limit=200, is_duplicate=True)
    assert {d.document_id for d in docs} == {"cloud-011", "sec-011"} and total == 2
    total, docs = fast_engine.list_documents(limit=200, q="kubernetes")
    assert docs and all("kubernetes" in d.title.lower() or "kubernetes" in d.document_id for d in docs)
    total, docs = fast_engine.list_documents(limit=200, is_outlier=True)
    assert total == sum(d.is_outlier for d in fast_engine.documents.values())
    some_cluster = next(d.cluster_id for d in fast_engine.documents.values() if d.cluster_id is not None)
    _, docs = fast_engine.list_documents(limit=200, cluster_id=some_cluster)
    assert docs and all(d.cluster_id == some_cluster for d in docs)
    _, docs = fast_engine.list_documents(limit=200, source="syndicated-feed")
    assert {d.document_id for d in docs} == {"cloud-011", "sec-011"}


def test_evaluation_searches_do_not_count_as_user_searches(fast_engine):
    before = fast_engine.analytics()["search_stats"]["total_searches"]
    fast_engine.search("tcp vs udp", record_stats=False)
    assert fast_engine.analytics()["search_stats"]["total_searches"] == before


def test_bm25_available_reflects_elasticsearch_outage(fast_engine):
    from unittest.mock import MagicMock

    from app.retrieval.elasticsearch_retriever import ElasticsearchRetriever

    assert fast_engine.bm25_available() is True  # in-memory backend
    client = MagicMock()
    client.ping.side_effect = ConnectionError("down")
    fast_engine.lexical = ElasticsearchRetriever(client, "docs")
    assert fast_engine.bm25_backend == "elasticsearch"
    assert fast_engine.bm25_available() is False
