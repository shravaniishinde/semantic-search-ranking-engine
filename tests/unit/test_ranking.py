import numpy as np
import pytest

from app.models.schemas import Document, Entity, HybridCandidate, IntentPrediction
from app.ranking.features import FEATURE_NAMES, QueryContext, RankingFeatureExtractor, category_match, to_matrix
from app.ranking.reranker import LTRDataset, XGBoostReranker

DOCS = {
    "aws": Document(
        document_id="aws",
        title="Secure architecture on AWS",
        category="cloud computing",
        content="IAM roles and KMS encryption for AWS workloads.",
        entities=[Entity(text="AWS", label="ORG")],
    ),
    "tcp": Document(
        document_id="tcp",
        title="TCP versus UDP",
        category="networking",
        content="Handshakes, reliability and latency trade-offs between transport protocols.",
    ),
}
VECTORS = {"aws": np.array([1.0, 0.0], dtype=np.float32), "tcp": np.array([0.0, 1.0], dtype=np.float32)}


def ctx(query="security architecture for AWS", vector=(1.0, 0.0), intent="technical"):
    return QueryContext(
        query=query,
        query_vector=np.array(vector, dtype=np.float32),
        entities=[Entity(text="AWS", label="ORG")] if "AWS" in query else [],
        intent=IntentPrediction(label=intent, confidence=0.8, source="xgboost"),
    )


CANDIDATES = [
    HybridCandidate(document_id="aws", hybrid_score=0.0325, bm25_score=6.0, bm25_rank=1, semantic_score=0.9, semantic_rank=1),
    HybridCandidate(document_id="tcp", hybrid_score=0.0160, semantic_score=0.1, semantic_rank=2),
]


def test_feature_rows_have_all_named_features():
    rows = RankingFeatureExtractor(candidate_depth=30).extract(ctx(), CANDIDATES, DOCS, VECTORS)
    assert len(rows) == 2
    assert all(list(r) == FEATURE_NAMES for r in rows)
    assert to_matrix(rows).shape == (2, len(FEATURE_NAMES))


def test_feature_values():
    aws, tcp = RankingFeatureExtractor(candidate_depth=30).extract(ctx(), CANDIDATES, DOCS, VECTORS)
    assert aws["bm25_score_norm"] == 1.0 and aws["in_bm25"] == 1.0
    assert tcp["bm25_rank"] == 31.0 and tcp["in_bm25"] == 0.0 and tcp["bm25_score"] == 0.0  # missing -> depth+1
    assert aws["semantic_similarity"] == pytest.approx(1.0) and tcp["semantic_similarity"] == pytest.approx(0.0)
    assert aws["entity_overlap"] == 1.0 and tcp["entity_overlap"] == 0.0
    # query tokens: security, architecture, aws; the doc says "secure", not "security"
    assert aws["query_term_coverage"] == pytest.approx(2 / 3)
    assert aws["title_term_coverage"] == pytest.approx(2 / 3)
    assert aws["category_match"] == 1.0 and tcp["category_match"] == 0.0
    assert aws["intent_technical"] == 1.0 and aws["intent_comparison"] == 0.0
    assert aws["intent_confidence"] == 0.8
    assert aws["hybrid_rank"] == 1.0 and tcp["hybrid_rank"] == 2.0


def test_title_phrase_match():
    rows = RankingFeatureExtractor().extract(ctx("TCP versus UDP", (0, 1), "comparison"), CANDIDATES, DOCS, VECTORS)
    assert [r["title_phrase_match"] for r in rows] == [0.0, 1.0]


def test_category_match_uses_vocabulary():
    assert category_match({"sourdough"}, "devops") == 0.0
    assert category_match({"kubectl"}, "devops") == 1.0
    assert category_match({"kubernetes"}, "devops") == 1.0
    assert category_match({"bill"}, "cloud computing") == 1.0


def test_untrained_reranker_keeps_hybrid_order():
    rows = RankingFeatureExtractor().extract(ctx(), CANDIDATES, DOCS, VECTORS)
    ranked = XGBoostReranker().rerank(CANDIDATES, rows)
    assert [r.document_id for r in ranked] == ["aws", "tcp"]
    assert ranked[0].ranker == "hybrid_fallback"
    assert ranked[0].final_score == pytest.approx(CANDIDATES[0].hybrid_score)
    assert ranked[0].features == rows[0] and ranked[1].bm25_rank is None


def _synthetic_dataset(rng, n_queries=40, per_query=10):
    """Relevance is driven by entity_overlap and title coverage; hybrid order is random noise."""
    rows, labels = [], []
    for _ in range(n_queries):
        for i in range(per_query):
            row = {name: float(rng.random()) for name in FEATURE_NAMES}
            row["hybrid_rank"] = float(i + 1)
            relevant = rng.random() < 0.25
            row["entity_overlap"] = 1.0 if relevant else 0.0
            row["title_term_coverage"] = rng.uniform(0.6, 1.0) if relevant else rng.uniform(0.0, 0.4)
            rows.append(row)
            labels.append(2.0 if relevant else 0.0)
    return LTRDataset(to_matrix(rows), np.array(labels, dtype=np.float32), [per_query] * n_queries, []), rows


def test_trained_reranker_learns_relevance_signal():
    rng = np.random.default_rng(0)
    dataset, _ = _synthetic_dataset(rng)
    reranker = XGBoostReranker().fit(dataset)
    assert reranker.is_trained and reranker.name == "xgboost_ltr"
    importance = reranker.feature_importance()
    assert set(list(importance)[:2]) & {"entity_overlap", "title_term_coverage"}

    # Fresh query: relevant document sits last in the hybrid order and must move to the top.
    _, test_rows = _synthetic_dataset(np.random.default_rng(1), n_queries=1, per_query=5)
    for r in test_rows:
        r["entity_overlap"], r["title_term_coverage"] = 0.0, 0.1
    test_rows[-1]["entity_overlap"], test_rows[-1]["title_term_coverage"] = 1.0, 0.95
    candidates = [HybridCandidate(document_id=f"d{i}", hybrid_score=1 / (60 + i)) for i in range(5)]
    ranked = reranker.rerank(candidates, test_rows)
    assert ranked[0].document_id == "d4"
    assert [r.rank for r in ranked] == [1, 2, 3, 4, 5]


def test_reranker_save_load_and_alignment_errors(tmp_path):
    dataset, rows = _synthetic_dataset(np.random.default_rng(2), n_queries=10)
    reranker = XGBoostReranker().fit(dataset)
    path = tmp_path / "r.joblib"
    reranker.save(path)
    restored = XGBoostReranker()
    assert restored.load(path)
    assert np.allclose(restored.score(rows[:5]), reranker.score(rows[:5]))
    with pytest.raises(ValueError):
        reranker.rerank(CANDIDATES, rows[:1])
    with pytest.raises(ValueError):
        XGBoostReranker().fit(LTRDataset(np.zeros((0, len(FEATURE_NAMES))), np.zeros(0), [], []))
