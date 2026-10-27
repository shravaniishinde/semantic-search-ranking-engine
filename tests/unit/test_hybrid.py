import pytest

from app.embeddings.encoder import HashingEmbedder
from app.models.schemas import RetrievalResult
from app.retrieval.faiss_retriever import FaissIndex, FaissRetriever
from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def results(method, ids, scores=None):
    scores = scores or [10.0 - i for i in range(len(ids))]
    return [
        RetrievalResult(document_id=d, score=s, rank=i + 1, retrieval_method=method) for i, (d, s) in enumerate(zip(ids, scores))
    ]


def test_bm25_only_results():
    fused = reciprocal_rank_fusion(results("bm25", ["a", "b"]), [], k=60)
    assert [c.document_id for c in fused] == ["a", "b"]
    assert fused[0].hybrid_score == pytest.approx(1 / 61)
    assert fused[0].semantic_rank is None and fused[0].bm25_rank == 1
    assert fused[0].sources == ["bm25"]


def test_semantic_only_results():
    fused = reciprocal_rank_fusion([], results("semantic", ["x", "y"], [0.9, 0.5]), k=10)
    assert [c.document_id for c in fused] == ["x", "y"]
    assert fused[1].hybrid_score == pytest.approx(1 / 12)
    assert fused[0].semantic_score == 0.9 and fused[0].bm25_score is None


def test_documents_in_both_lists_are_merged_and_boosted():
    bm25 = results("bm25", ["a", "shared", "b"])
    semantic = results("semantic", ["c", "shared", "d"], [0.9, 0.8, 0.7])
    fused = reciprocal_rank_fusion(bm25, semantic, k=60)
    ids = [c.document_id for c in fused]
    assert ids.count("shared") == 1
    assert ids[0] == "shared"  # 2/(62) beats 1/61
    shared = fused[0]
    assert (shared.bm25_rank, shared.semantic_rank) == (2, 2)
    assert shared.bm25_score == 9.0 and shared.semantic_score == 0.8
    assert shared.hybrid_score == pytest.approx(2 / 62)
    assert sorted(shared.sources) == ["bm25", "semantic"]
    assert len(fused) == 5


def test_duplicate_ids_within_one_list_keep_best_rank():
    bm25 = results("bm25", ["a", "a", "b"])
    fused = reciprocal_rank_fusion(bm25, [], k=60)
    a = next(c for c in fused if c.document_id == "a")
    assert a.bm25_rank == 1 and a.hybrid_score == pytest.approx(1 / 61)


def test_weights_and_top_k():
    fused = reciprocal_rank_fusion(results("bm25", ["a"]), results("semantic", ["b"]), k=60, semantic_weight=2.0, top_k=1)
    assert [c.document_id for c in fused] == ["b"]


def test_rrf_is_rank_based_not_score_based():
    # Huge BM25 scores must not dominate.
    fused = reciprocal_rank_fusion(results("bm25", ["a"], [1000.0]), results("semantic", ["b"], [0.1]), k=60)
    assert fused[0].hybrid_score == fused[1].hybrid_score


class _FailingLexical:
    method = "bm25"

    def retrieve(self, query, top_k, filters=None):
        raise ConnectionError("elasticsearch down")


class _StaticLexical:
    method = "bm25"

    def __init__(self, ids):
        self.ids = ids
        self.requested_depth = None

    def retrieve(self, query, top_k, filters=None):
        self.requested_depth = top_k
        return results("bm25", self.ids[:top_k])


def _semantic():
    emb = HashingEmbedder(64)
    idx = FaissIndex(64)
    idx.add(["s1", "s2"], emb.encode(["kubernetes pods", "database index"]))
    return FaissRetriever(emb, idx)


def test_hybrid_retriever_fetches_candidate_depth_and_fuses():
    lexical = _StaticLexical(["s2", "l1"])
    bm25, semantic, fused = HybridRetriever(lexical, _semantic(), candidate_k=25).retrieve_all("database index", 5)
    assert lexical.requested_depth == 25
    assert fused[0].document_id == "s2"  # top of both lists
    assert {c.document_id for c in fused} == {"s1", "s2", "l1"}


def test_hybrid_degrades_to_semantic_when_lexical_backend_fails():
    hybrid = HybridRetriever(_FailingLexical(), _semantic())
    bm25, semantic, fused = hybrid.retrieve_all("kubernetes", 5)
    assert bm25 == [] and len(semantic) == 2
    assert all(c.bm25_rank is None for c in fused)
    assert hybrid.last_lexical_error == "ConnectionError"
