"""Dense retrieval with FAISS.

Vectors are L2-normalised and stored in an exact ``IndexFlatIP``, so inner product equals cosine
similarity. Exact search is the right call at this corpus size; swapping in ``IndexHNSWFlat``
would be a one-line change for larger corpora.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np

from app.embeddings.encoder import Embedder, l2_normalize
from app.models.schemas import RetrievalResult, SearchFilters

logger = logging.getLogger(__name__)


class FaissIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.doc_ids: list[str] = []  # FAISS position -> document_id
        self._positions: dict[str, int] = {}

    def __len__(self) -> int:
        return self.index.ntotal

    def __contains__(self, document_id: str) -> bool:
        return document_id in self._positions

    def add(self, document_ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(l2_normalize(vectors))
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"expected vectors of shape (n, {self.dimension}), got {vectors.shape}")
        if len(document_ids) != len(vectors):
            raise ValueError("document_ids and vectors must have the same length")
        duplicates = {d for d in document_ids if d in self._positions}
        if len(set(document_ids)) != len(document_ids):
            duplicates |= {d for d in document_ids if document_ids.count(d) > 1}
        if duplicates:
            raise ValueError(f"documents already indexed: {sorted(duplicates)[:5]}")
        start = len(self.doc_ids)
        self.index.add(vectors)
        for offset, doc_id in enumerate(document_ids):
            self.doc_ids.append(doc_id)
            self._positions[doc_id] = start + offset

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if len(self) == 0 or top_k <= 0:
            return []
        query = np.ascontiguousarray(l2_normalize(query_vector.reshape(1, -1)))
        scores, positions = self.index.search(query, min(top_k, len(self)))
        return [(self.doc_ids[p], float(s)) for p, s in zip(positions[0], scores[0]) if p != -1]

    def vector(self, document_id: str) -> np.ndarray | None:
        position = self._positions.get(document_id)
        return None if position is None else self.index.reconstruct(position)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        path.with_suffix(".ids.json").write_text(json.dumps(self.doc_ids), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FaissIndex:
        index = faiss.read_index(str(path))
        doc_ids = json.loads(path.with_suffix(".ids.json").read_text(encoding="utf-8"))
        if len(doc_ids) != index.ntotal:
            raise ValueError("FAISS index and id mapping are out of sync")
        obj = cls(index.d)
        obj.index = index
        obj.doc_ids = doc_ids
        obj._positions = {d: i for i, d in enumerate(doc_ids)}
        return obj


class FaissRetriever:
    method = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        index: FaissIndex,
        filter_fn: Callable[[str, SearchFilters], bool] | None = None,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.filter_fn = filter_fn  # document_id -> passes filters

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: SearchFilters | None = None,
        query_vector: np.ndarray | None = None,
    ) -> list[RetrievalResult]:
        if query_vector is None:
            query_vector = self.embedder.encode_query(query)
        # Over-fetch when filtering so post-filtering still yields top_k hits.
        fetch = len(self.index) if filters and self.filter_fn else top_k
        hits = self.index.search(query_vector, fetch)
        if filters and self.filter_fn:
            hits = [(d, s) for d, s in hits if self.filter_fn(d, filters)]
        return [
            RetrievalResult(document_id=d, score=round(s, 6), rank=r, retrieval_method=self.method)
            for r, (d, s) in enumerate(hits[:top_k], start=1)
        ]
