"""End-to-end tests with the real sentence-transformer, spaCy and trained XGBoost models."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.evaluation.evaluator import SYSTEMS, evaluate, load_qrels
from app.retrieval.bm25_memory import InMemoryBM25Retriever
from app.services.search_service import SearchEngine

pytestmark = pytest.mark.integration


def test_real_embeddings_are_normalized_and_semantic(real_engine):
    emb = real_engine.embedder
    a, b, c = emb.encode(
        [
            "How do I stop brute-force login attacks?",
            "Rate limiting failed sign-in attempts protects accounts.",
            "Sourdough bread needs an active starter.",
        ]
    )
    assert a.shape == (384,)
    assert abs(float(a @ a) - 1.0) < 1e-4
    assert float(a @ b) > float(a @ c) + 0.2


def test_semantic_retrieval_handles_paraphrase_better_than_bm25(real_engine):
    query = "stop attackers from guessing user passwords"
    bm25, semantic, _ = real_engine.hybrid.retrieve_all(query, 10)
    sem_rank = next(r.rank for r in semantic if r.document_id == "sec-002")
    bm25_rank = next((r.rank for r in bm25 if r.document_id == "sec-002"), None)
    assert sem_rank == 1
    assert bm25_rank is None or bm25_rank > sem_rank


def test_bm25_nails_exact_identifiers(real_engine):
    bm25, _, _ = real_engine.hybrid.retrieve_all("CVE-2021-44228", 10)
    assert bm25[0].document_id == "sec-001"


def test_dbscan_flags_sample_duplicates_and_outliers(real_engine):
    docs = real_engine.documents
    assert docs["cloud-011"].is_duplicate and docs["cloud-011"].duplicate_of == "cloud-006"
    assert docs["sec-011"].is_duplicate and docs["sec-011"].duplicate_of == "sec-001"
    # The paraphrased container article is semantically close but not a confirmed duplicate.
    assert not docs["devops-011"].is_duplicate and docs["devops-011"].indexed
    for off_topic in ("misc-001", "misc-002", "misc-003"):
        assert docs[off_topic].is_outlier
    analysis = real_engine.analysis
    assert analysis.cluster_count >= 3
    assert analysis.confirmed_duplicate_count == 2


@pytest.mark.parametrize(
    "query, expected_doc, expected_intent",
    [
        ("compare TCP and UDP", "net-001", "comparison"),
        ("why is my kubernetes pod restarting repeatedly", "devops-004", "troubleshooting"),
        ("security architecture for AWS", "cloud-001", None),
    ],
)
def test_complete_search_flow(real_engine, query, expected_doc, expected_intent):
    r = real_engine.search(query, top_k=5)
    assert r.ranker == "xgboost_ltr"
    # The highly relevant document must be near the top (exact order among relevant docs is
    # left to the learned ranker; e.g. the generic Kubernetes article can outrank CrashLoopBackOff).
    assert expected_doc in [i.document_id for i in r.results[:2]]
    if expected_intent:
        assert r.intent.label == expected_intent and r.intent.source == "xgboost"
    top = next(i for i in r.results if i.document_id == expected_doc)
    assert top.features["semantic_similarity"] > 0.3


def test_query_entities_exposed(real_engine):
    r = real_engine.search("security architecture for AWS", top_k=3)
    assert ("AWS", "ORG") in {(e.text, e.label) for e in r.entities}


def test_evaluation_reports_all_systems(real_engine, tmp_path):
    from app.config.settings import PROJECT_ROOT

    qrels = load_qrels(PROJECT_ROOT / "data" / "sample" / "evaluation_qrels.json")
    report = evaluate(real_engine, qrels)
    assert set(report["metrics"]) == set(SYSTEMS)
    for metrics in report["metrics"].values():
        assert set(metrics) == {"MRR", "P@5", "R@5", "NDCG@5", "P@10", "R@10", "NDCG@10"}
        assert all(0.0 <= v <= 1.0 for v in metrics.values())
    # Hybrid should not be worse than the weaker single retriever on recall.
    m = report["metrics"]
    assert m["hybrid"]["R@10"] >= min(m["bm25"]["R@10"], m["semantic"]["R@10"])


def test_state_persists_and_reloads(real_engine):
    expected = [r.document_id for r in real_engine.search("Terraform state file").results]
    restored = SearchEngine(real_engine.settings, entity_extractor=real_engine.ner, lexical=InMemoryBM25Retriever())
    restored.embedder = real_engine.embedder  # reuse the loaded model
    assert restored.load_state()
    restored.ensure_models(train_if_missing=False)
    assert restored.reranker.is_trained
    assert [r.document_id for r in restored.search("Terraform state file").results] == expected


def test_api_with_real_engine(real_engine):
    with TestClient(create_app(engine_factory=lambda s: real_engine, settings=real_engine.settings)) as client:
        body = client.post("/search", json={"query": "reduce cloud bill", "top_k": 3}).json()
        assert body["results"][0]["document_id"] == "cloud-005"
        assert client.get("/clusters").json()["confirmed_duplicate_count"] == 2
