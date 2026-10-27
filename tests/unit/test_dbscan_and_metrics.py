from datetime import date

import numpy as np
import pytest

from app.clustering.dbscan_analysis import DBSCANConfig, DocumentClusterAnalyzer, jaccard
from app.evaluation.metrics import evaluate_run, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from app.models.schemas import Document


def _doc(doc_id, text, day=1, category="c"):
    return Document(document_id=doc_id, title="", content=text, category=category, created_at=date(2024, 1, day))


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    a_center, b_center = np.eye(8)[0], np.eye(8)[1]
    docs, vecs = [], []
    for i in range(4):
        docs.append(_doc(f"a{i}", f"cloud storage topic {i} unique{i}", day=i + 1, category="cloud"))
        vecs.append(_unit(a_center + rng.normal(0, 0.15, 8)))
    for i in range(4):
        docs.append(_doc(f"b{i}", f"network routing topic {i} special{i}", day=i + 1, category="net"))
        vecs.append(_unit(b_center + rng.normal(0, 0.15, 8)))
    docs.append(_doc("outlier", "sourdough bread recipe"))
    vecs.append(_unit(np.eye(8)[5]))
    # Exact copy of a0 published later -> confirmed duplicate.
    docs.append(_doc("a0_copy", docs[0].content, day=20, category="cloud"))
    vecs.append(vecs[0].copy())
    # Near-identical vector but different wording -> candidate only.
    docs.append(_doc("b0_para", "entirely different wording here", day=21, category="net"))
    vecs.append(_unit(vecs[4] + 0.05 * np.eye(8)[2]))
    return docs, np.stack(vecs)


def test_clusters_outliers_and_duplicates(corpus):
    docs, vecs = corpus
    analysis = DocumentClusterAnalyzer(DBSCANConfig(eps=0.2, min_samples=3, center=False)).analyze(docs, vecs)
    by_id = {d.document_id: d for d in docs}

    assert analysis.cluster_count == 2
    assert analysis.outliers == ["outlier"]
    assert by_id["outlier"].is_outlier and by_id["outlier"].cluster_id == -1
    assert by_id["a1"].cluster_id == by_id["a2"].cluster_id != by_id["b1"].cluster_id
    assert {c.dominant_category for c in analysis.clusters} == {"cloud", "net"}

    # Confirmed duplicate: later copy points at the earlier canonical doc; nothing is deleted.
    assert by_id["a0_copy"].is_duplicate and by_id["a0_copy"].duplicate_of == "a0"
    assert not by_id["a0"].is_duplicate and by_id["a0"].duplicate_candidate
    # Paraphrase: similar embedding but low lexical overlap -> candidate, not confirmed.
    assert by_id["b0_para"].duplicate_candidate and not by_id["b0_para"].is_duplicate
    pair = next(p for p in analysis.duplicate_pairs if p.other_document_id == "b0_para")
    assert not pair.confirmed and pair.jaccard_similarity < 0.5
    assert analysis.confirmed_duplicate_count == 1
    assert analysis.total_documents == len(docs)


def test_reanalysis_resets_flags(corpus):
    docs, vecs = corpus
    analyzer = DocumentClusterAnalyzer(DBSCANConfig(eps=0.2, min_samples=3, center=False))
    analyzer.analyze(docs, vecs)
    analyzer.config.duplicate_candidate_threshold = 1.01  # disable duplicate detection
    analysis = analyzer.analyze(docs, vecs)
    assert analysis.confirmed_duplicate_count == 0 and not any(d.is_duplicate for d in docs)


def test_mean_centering_separates_topics_sharing_a_common_component():
    """Sentence embeddings share a large common direction; centring removes it before DBSCAN."""
    rng = np.random.default_rng(3)
    shared = np.eye(8)[7] * 3.0
    vecs = np.stack([_unit(shared + np.eye(8)[axis] + rng.normal(0, 0.05, 8)) for axis in (0, 0, 0, 1, 1, 1)])
    docs = [_doc(f"d{i}", f"text {i}") for i in range(6)]
    raw = DocumentClusterAnalyzer(DBSCANConfig(eps=0.1, min_samples=2, center=False)).analyze(docs, vecs)
    centred = DocumentClusterAnalyzer(DBSCANConfig(eps=0.1, min_samples=2, center=True)).analyze(docs, vecs)
    assert raw.cluster_count == 1
    assert centred.cluster_count == 2 and centred.outlier_count == 0


def test_empty_corpus():
    analysis = DocumentClusterAnalyzer().analyze([], np.zeros((0, 4)))
    assert analysis.total_documents == 0 and analysis.clusters == []


def test_jaccard():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 1.0


# ------------------------------------------------------------------ metrics
REL = {"d1": 2, "d3": 1}


def test_precision_recall():
    ranked = ["d1", "d2", "d3", "d4"]
    assert precision_at_k(ranked, REL, 2) == 0.5
    assert precision_at_k(ranked, REL, 4) == 0.5
    assert recall_at_k(ranked, REL, 1) == 0.5
    assert recall_at_k(ranked, REL, 3) == 1.0
    assert recall_at_k(ranked, {}, 3) == 0.0


def test_mrr():
    assert reciprocal_rank(["x", "d3"], REL) == 0.5
    assert reciprocal_rank(["x", "y"], REL) == 0.0


def test_ndcg_known_values():
    assert ndcg_at_k(["d1", "d3"], REL, 2) == pytest.approx(1.0)
    # DCG = (2^1-1)/log2(2) + (2^2-1)/log2(3); IDCG = 3/1 + 1/log2(3)
    expected = (1 + 3 / np.log2(3)) / (3 + 1 / np.log2(3))
    assert ndcg_at_k(["d3", "d1"], REL, 2) == pytest.approx(expected)
    assert ndcg_at_k(["x"], {}, 5) == 0.0


def test_evaluate_run_averages_over_queries():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    runs = {"q1": ["a", "x"], "q2": ["x", "b"]}
    metrics = evaluate_run(runs, qrels, ks=(1,))
    assert metrics["MRR"] == 0.75
    assert metrics["P@1"] == 0.5
