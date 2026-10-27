"""Hybrid retrieval with Reciprocal Rank Fusion (RRF).

BM25 scores are unbounded and corpus-dependent while cosine similarities live in [-1, 1], so
adding raw scores would let one retriever dominate. RRF uses ranks only:

    RRF(d) = sum_r  w_r / (k + rank_r(d))

Documents found by both retrievers are merged into one candidate that keeps both raw scores and
ranks; those become features for the reranker.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from app.models.schemas import HybridCandidate, RetrievalResult, SearchFilters
from app.retrieval.base import Retriever
from app.retrieval.faiss_retriever import FaissRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: list[RetrievalResult],
    semantic_results: list[RetrievalResult],
    k: int = 60,
    bm25_weight: float = 1.0,
    semantic_weight: float = 1.0,
    top_k: int | None = None,
) -> list[HybridCandidate]:
    candidates: dict[str, HybridCandidate] = {}

    def _candidate(doc_id: str) -> HybridCandidate:
        if doc_id not in candidates:
            candidates[doc_id] = HybridCandidate(document_id=doc_id, hybrid_score=0.0)
        return candidates[doc_id]

    for result in bm25_results:
        cand = _candidate(result.document_id)
        if cand.bm25_rank is not None:  # duplicate id within one list: keep the best rank
            continue
        cand.bm25_score, cand.bm25_rank = result.score, result.rank
        cand.hybrid_score += bm25_weight / (k + result.rank)
        cand.sources.append("bm25")
    for result in semantic_results:
        cand = _candidate(result.document_id)
        if cand.semantic_rank is not None:
            continue
        cand.semantic_score, cand.semantic_rank = result.score, result.rank
        cand.hybrid_score += semantic_weight / (k + result.rank)
        cand.sources.append("semantic")

    fused = sorted(candidates.values(), key=lambda c: (-c.hybrid_score, c.document_id))
    for cand in fused:
        cand.hybrid_score = round(cand.hybrid_score, 8)
    return fused[:top_k] if top_k else fused


class HybridRetriever:
    method = "hybrid"

    def __init__(
        self,
        lexical: Retriever,
        semantic: FaissRetriever,
        rrf_k: int = 60,
        candidate_k: int = 30,
        bm25_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> None:
        self.lexical = lexical
        self.semantic = semantic
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.last_lexical_error: str | None = None
        self.last_timings_ms: dict[str, float] = {}

    def retrieve_all(
        self,
        query: str,
        top_k: int,
        filters: SearchFilters | None = None,
        query_vector: np.ndarray | None = None,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult], list[HybridCandidate]]:
        depth = max(top_k, self.candidate_k)
        t0 = time.perf_counter()
        # A lexical backend outage (e.g. Elasticsearch down) degrades to semantic-only instead of failing.
        try:
            bm25 = self.lexical.retrieve(query, depth, filters)
            self.last_lexical_error = None
        except Exception as exc:  # noqa: BLE001
            logger.error("retrieval.bm25_failed", extra={"fields": {"error": str(exc)[:200]}})
            bm25, self.last_lexical_error = [], type(exc).__name__
        t1 = time.perf_counter()
        semantic = self.semantic.retrieve(query, depth, filters, query_vector=query_vector)
        t2 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            bm25, semantic, k=self.rrf_k, bm25_weight=self.bm25_weight, semantic_weight=self.semantic_weight, top_k=top_k
        )
        t3 = time.perf_counter()
        self.last_timings_ms = {
            "bm25": round((t1 - t0) * 1000, 2),
            "semantic": round((t2 - t1) * 1000, 2),
            "fusion": round((t3 - t2) * 1000, 2),
        }
        return bm25, semantic, fused

    def retrieve(self, query: str, top_k: int, filters: SearchFilters | None = None) -> list[HybridCandidate]:
        return self.retrieve_all(query, top_k, filters)[2]
