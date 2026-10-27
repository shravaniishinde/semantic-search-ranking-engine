"""API contract tests against an engine built from lightweight components."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.embeddings.encoder import HashingEmbedder
from app.retrieval.bm25_memory import InMemoryBM25Retriever
from app.services.search_service import SearchEngine
from tests.conftest import make_settings


@pytest.fixture(scope="module")
def client(tmp_path_factory, ner):
    settings = make_settings(tmp_path_factory.mktemp("api"))

    def factory(s):
        return SearchEngine(s, embedder=HashingEmbedder(), entity_extractor=ner, lexical=InMemoryBM25Retriever(), persist=False)

    with TestClient(create_app(engine_factory=factory, settings=settings)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["bm25_backend"] == "memory"
    assert body["documents"] > 70 and body["reranker"] == "xgboost_ltr"


def test_search_contract(client):
    resp = client.post("/search", json={"query": "security architecture for AWS", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"query", "intent", "entities", "retrieval", "results", "latency_ms"}
    assert {"label", "confidence"} <= set(body["intent"])
    assert any(e["text"] == "AWS" for e in body["entities"])
    assert set(body["retrieval"]) >= {"bm25", "semantic", "hybrid"}
    assert len(body["results"]) == 3
    item = body["results"][0]
    for key in ("document_id", "title", "final_score", "bm25_score", "semantic_score", "hybrid_score", "features", "rank"):
        assert key in item
    assert item["rank"] == 1


def test_search_without_retrieval_debug(client):
    body = client.post("/search", json={"query": "redis cache", "include_retrieval": False}).json()
    assert body["retrieval"] is None and body["results"]


@pytest.mark.parametrize(
    "payload, status", [({"query": ""}, 422), ({"query": "ok", "top_k": 500}, 422), ({"query": "<p></p>"}, 400), ({}, 422)]
)
def test_search_validation(client, payload, status):
    assert client.post("/search", json=payload).status_code == status


def test_documents_listing_and_lookup(client):
    page = client.get("/documents", params={"limit": 5, "offset": 0}).json()
    assert page["limit"] == 5 and len(page["documents"]) == 5 and page["total"] > 70
    filtered = client.get("/documents", params={"category": "networking", "limit": 50}).json()
    assert filtered["total"] == 10 and all(d["category"] == "networking" for d in filtered["documents"])
    only_indexed = client.get("/documents", params={"include_excluded": False, "limit": 200}).json()
    assert all(d["indexed"] for d in only_indexed["documents"])

    doc = client.get("/documents/db-008").json()
    assert doc["title"] == "Troubleshooting MySQL Deadlocks" and "entities" in doc and "cluster_id" in doc
    assert client.get("/documents/nope").status_code == 404


def test_analytics_and_clusters(client):
    a = client.get("/analytics").json()
    for key in ("document_count", "indexed_count", "cluster_count", "duplicate_count", "outlier_count", "search_stats"):
        assert key in a
    c = client.get("/clusters").json()
    assert c["total_documents"] == a["document_count"]
    assert c["cluster_count"] == len(c["clusters"])
    assert {"duplicate_pairs", "outliers"} <= set(c)


def test_ingest_endpoint_makes_documents_searchable(client):
    payload = {
        "documents": [
            {
                "document_id": "new-1",
                "title": "Quantum key distribution",
                "content": "QKD uses photons to exchange keys securely between parties.",
                "category": "cybersecurity",
            },
            {"document_id": "new-2", "title": "Empty", "content": " "},
        ]
    }
    report = client.post("/documents/ingest", json=payload).json()
    assert report["accepted"] == 1 and report["rejected"][0]["document_id"] == "new-2"
    hits = client.post("/search", json={"query": "quantum key distribution photons"}).json()["results"]
    assert hits[0]["document_id"] == "new-1"
    assert client.post("/documents/ingest", json={"documents": []}).status_code == 422


def test_health_reports_backend_details(client):
    body = client.get("/health").json()
    assert body["bm25_backend_requested"] == "memory"
    assert body["bm25_available"] is True
    assert body["faiss_vectors"] == body["indexed"] and body["bm25_documents"] == body["indexed"]
    assert body["embedding_model"].startswith("hashing-bow")


def test_cors_allows_dashboard_origin(client):
    resp = client.options("/search", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
    blocked = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in blocked.headers


def test_search_exposes_retrieval_counts_and_stage_breakdown(client):
    body = client.post("/search", json={"query": "database replication lag", "top_k": 3}).json()
    assert set(body["retrieval_counts"]) == {"bm25", "semantic", "hybrid"}
    assert body["retrieval_counts"]["hybrid"] == body["candidate_count"]
    assert {"bm25", "semantic", "fusion", "rerank"} <= set(body["stage_latency_ms"])


def test_documents_endpoint_filters(client):
    dups = client.get("/documents", params={"is_duplicate": True, "limit": 50}).json()
    assert {d["document_id"] for d in dups["documents"]} == {"cloud-011", "sec-011"}
    found = client.get("/documents", params={"q": "DEADLOCK"}).json()
    assert [d["document_id"] for d in found["documents"]] == ["db-008"]
    outliers = client.get("/documents", params={"is_outlier": True, "limit": 200}).json()
    assert outliers["total"] == client.get("/analytics").json()["outlier_count"]
    assert "sources" in client.get("/analytics").json()


def test_evaluation_endpoints(client):
    assert client.get("/evaluation").json() == {"available": False, "report": None}
    run = client.post("/evaluation/run").json()
    assert run["available"] is True
    report = run["report"]
    assert set(report["metrics"]) == {"bm25", "semantic", "hybrid", "hybrid+rerank"}
    assert report["queries"] == 20 and report["ks"] == [5, 10]
    assert client.get("/evaluation").json()["report"]["generated_at"] == report["generated_at"]
