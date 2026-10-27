"""Text embedding models and an on-disk embedding cache.

``SentenceTransformerEmbedder`` is the real semantic encoder (pretrained, not fine-tuned).
``HashingEmbedder`` is a deterministic bag-of-words stand-in used by fast unit tests; it has no
semantic understanding and is never used by the running service.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from app.preprocessing.text import tokenize

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu") -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._dimension: int | None = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("embeddings.model_loading", extra={"fields": {"model": self.model_name}})
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
            self._dimension = int(getter())
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEmbedder:
    """Deterministic hashed bag-of-words vectors (test double, not semantic)."""

    def __init__(self, dimension: int = 256) -> None:
        self.model_name = f"hashing-bow-{dimension}"
        self.dimension = dimension

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.md5(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "little") % self.dimension] += 1.0
        if not vec.any():
            vec[0] = 1.0
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def content_hash(model_name: str, text: str) -> str:
    return hashlib.sha1(f"{model_name}\x00{text}".encode()).hexdigest()


class EmbeddingCache:
    """Maps content hashes to vectors so unchanged documents are never re-encoded."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._store: dict[str, np.ndarray] = {}
        if path is not None and path.exists():
            with np.load(path) as data:
                self._store = {key: data[key] for key in data.files}

    def __len__(self) -> int:
        return len(self._store)

    def get(self, key: str) -> np.ndarray | None:
        return self._store.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._store[key] = np.asarray(vector, dtype=np.float32)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self._store)


def embed_with_cache(embedder: Embedder, texts: list[str], cache: EmbeddingCache) -> tuple[np.ndarray, list[str], int]:
    """Embed ``texts`` reusing cached vectors. Returns (vectors, hashes, newly_encoded_count)."""
    hashes = [content_hash(embedder.model_name, t) for t in texts]
    missing = [i for i, h in enumerate(hashes) if cache.get(h) is None]
    if missing:
        fresh = embedder.encode([texts[i] for i in missing])
        for i, vec in zip(missing, fresh):
            cache.put(hashes[i], vec)
    if not texts:
        return np.zeros((0, embedder.dimension), dtype=np.float32), hashes, 0
    vectors = np.stack([cache.get(h) for h in hashes]).astype(np.float32)
    return vectors, hashes, len(missing)
