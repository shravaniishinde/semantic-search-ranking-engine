import numpy as np

from app.embeddings.encoder import EmbeddingCache, HashingEmbedder, embed_with_cache, l2_normalize


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__(dimension=32)
        self.encoded = 0

    def encode(self, texts):
        self.encoded += len(texts)
        return super().encode(texts)


def test_l2_normalize_produces_unit_vectors_and_handles_zero():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.all(np.isfinite(out))


def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(64)
    a = emb.encode(["database replication lag"])
    b = emb.encode(["database replication lag"])
    assert a.shape == (1, 64)
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert emb.encode([]).shape == (0, 64)


def test_cache_avoids_recomputing_embeddings():
    emb, cache = CountingEmbedder(), EmbeddingCache()
    v1, _, new1 = embed_with_cache(emb, ["a b c", "d e f"], cache)
    v2, _, new2 = embed_with_cache(emb, ["d e f", "a b c", "g h i"], cache)
    assert (new1, new2) == (2, 1)
    assert emb.encoded == 3
    assert np.allclose(v1[0], v2[1])


def test_cache_persists_to_disk(tmp_path):
    path = tmp_path / "cache.npz"
    emb = CountingEmbedder()
    cache = EmbeddingCache(path)
    vectors, hashes, _ = embed_with_cache(emb, ["persist me"], cache)
    cache.save()
    reloaded = EmbeddingCache(path)
    assert len(reloaded) == 1
    assert np.allclose(reloaded.get(hashes[0]), vectors[0])


def test_cache_key_depends_on_model_name():
    cache = EmbeddingCache()
    embed_with_cache(HashingEmbedder(32), ["text"], cache)
    _, _, new = embed_with_cache(HashingEmbedder(16), ["text"], cache)
    assert new == 1
