import numpy as np
import pytest

from app.embeddings.encoder import HashingEmbedder
from app.models.schemas import SearchFilters
from app.retrieval.faiss_retriever import FaissIndex, FaissRetriever

TEXTS = {
    "net": "tcp udp packets network latency",
    "db": "postgres index query database",
    "ops": "kubernetes pod container deployment",
}


@pytest.fixture
def embedder():
    return HashingEmbedder(128)


@pytest.fixture
def index(embedder):
    idx = FaissIndex(embedder.dimension)
    idx.add(list(TEXTS), embedder.encode(list(TEXTS.values())))
    return idx


def test_search_returns_most_similar_document_first(index, embedder):
    hits = index.search(embedder.encode_query("database query index"), top_k=3)
    assert hits[0][0] == "db"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)


def test_scores_equal_cosine_similarity(index, embedder):
    q = embedder.encode_query("kubernetes pod")
    doc_id, score = index.search(q, 1)[0]
    expected = float(np.dot(q, embedder.encode([TEXTS[doc_id]])[0]))
    assert score == pytest.approx(expected, abs=1e-5)


def test_top_k_larger_than_index_and_empty_index(index, embedder):
    assert len(index.search(embedder.encode_query("anything"), 50)) == 3
    assert FaissIndex(8).search(np.ones(8, dtype=np.float32), 5) == []


def test_position_mapping_and_vector_reconstruction(index, embedder):
    assert index.doc_ids == ["net", "db", "ops"]
    assert "db" in index and "missing" not in index
    assert np.allclose(index.vector("db"), embedder.encode([TEXTS["db"]])[0], atol=1e-6)


def test_add_rejects_duplicates_and_bad_shapes(index, embedder):
    with pytest.raises(ValueError, match="already indexed"):
        index.add(["db"], embedder.encode(["x"]))
    with pytest.raises(ValueError):
        index.add(["new"], np.ones((1, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        index.add(["a", "b"], embedder.encode(["only one"]))


def test_save_and_load_roundtrip(index, embedder, tmp_path):
    path = tmp_path / "faiss.index"
    index.save(path)
    loaded = FaissIndex.load(path)
    q = embedder.encode_query("network packets")
    assert loaded.search(q, 3) == index.search(q, 3)
    assert loaded.doc_ids == index.doc_ids


def test_retriever_returns_structured_results(index, embedder):
    results = FaissRetriever(embedder, index).retrieve("container deployment", top_k=2)
    assert [r.rank for r in results] == [1, 2]
    assert results[0].document_id == "ops"
    assert results[0].retrieval_method == "semantic"


def test_retriever_applies_filters_after_overfetching(index, embedder):
    categories = {"net": "networking", "db": "databases", "ops": "devops"}
    retriever = FaissRetriever(embedder, index, filter_fn=lambda d, f: categories[d] == f.category)
    results = retriever.retrieve("database query", top_k=1, filters=SearchFilters(category="networking"))
    assert [r.document_id for r in results] == ["net"]
