"""DBSCAN over document embeddings: topical clusters, near-duplicates and outliers.

Terminology (kept distinct on purpose):
* cluster member      - DBSCAN assigned the document to a dense region (cluster_id >= 0)
* outlier             - DBSCAN noise (cluster_id == -1): no dense neighbourhood at ``eps``
* duplicate candidate - another document has cosine similarity >= candidate threshold
* confirmed duplicate - candidate pair that also passes cosine >= confirm threshold AND
                        token-set Jaccard >= jaccard threshold (guards against paraphrases)

Clustering runs on mean-centred, re-normalised embeddings: sentence-transformer vectors share a
large common component, so raw cosine similarities between unrelated technical texts are all
moderately high and DBSCAN chains everything into one cluster. Centring removes that shared
direction. Duplicate checks use the raw (uncentred) cosine, which is what the thresholds mean.

Nothing is deleted. The later document of a confirmed pair is marked ``is_duplicate`` with
``duplicate_of`` pointing at the canonical one; the engine decides what to exclude from indexing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from app.embeddings.encoder import l2_normalize
from app.models.schemas import ClusterAnalysis, ClusterInfo, Document, DuplicatePair
from app.preprocessing.text import tokenize


@dataclass
class DBSCANConfig:
    eps: float = 0.75
    min_samples: int = 2
    center: bool = True
    duplicate_candidate_threshold: float = 0.90
    duplicate_confirm_threshold: float = 0.95
    duplicate_jaccard_threshold: float = 0.5


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _sort_key(doc: Document) -> tuple:
    # Earlier documents are canonical; undated ones sort last.
    return (doc.created_at is None, doc.created_at or "", doc.document_id)


class DocumentClusterAnalyzer:
    def __init__(self, config: DBSCANConfig | None = None) -> None:
        self.config = config or DBSCANConfig()

    def analyze(self, documents: list[Document], vectors: np.ndarray) -> ClusterAnalysis:
        """Annotates ``documents`` in place and returns summary statistics."""
        cfg = self.config
        n = len(documents)
        if n == 0:
            return ClusterAnalysis(
                total_documents=0,
                cluster_count=0,
                clustered_documents=0,
                outlier_count=0,
                duplicate_candidate_count=0,
                confirmed_duplicate_count=0,
                eps=cfg.eps,
                min_samples=cfg.min_samples,
                clusters=[],
                outliers=[],
                duplicate_pairs=[],
            )
        vectors = l2_normalize(vectors)
        space = l2_normalize(vectors - vectors.mean(axis=0)) if cfg.center and n > 2 else vectors
        labels = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples, metric="cosine", algorithm="brute").fit_predict(space)

        for doc, label in zip(documents, labels):
            doc.cluster_id = int(label)
            doc.is_outlier = bool(label == -1)
            doc.duplicate_candidate = False
            doc.is_duplicate = False
            doc.duplicate_of = None

        pairs = self._duplicate_pairs(documents, vectors)
        clusters = self._cluster_summaries(documents)
        outliers = [d.document_id for d in documents if d.is_outlier]
        return ClusterAnalysis(
            total_documents=n,
            cluster_count=len(clusters),
            clustered_documents=n - len(outliers),
            outlier_count=len(outliers),
            duplicate_candidate_count=sum(d.duplicate_candidate for d in documents),
            confirmed_duplicate_count=sum(d.is_duplicate for d in documents),
            eps=cfg.eps,
            min_samples=cfg.min_samples,
            clusters=clusters,
            outliers=outliers,
            duplicate_pairs=pairs,
        )

    def _duplicate_pairs(self, documents: list[Document], vectors: np.ndarray) -> list[DuplicatePair]:
        cfg = self.config
        sims = vectors @ vectors.T
        token_sets = [set(tokenize(d.full_text)) for d in documents]
        order = sorted(range(len(documents)), key=lambda i: _sort_key(documents[i]))
        pairs: list[DuplicatePair] = []
        for a_pos, i in enumerate(order):
            for j in order[a_pos + 1 :]:
                cosine = float(sims[i, j])
                if cosine < cfg.duplicate_candidate_threshold:
                    continue
                jac = jaccard(token_sets[i], token_sets[j])
                confirmed = cosine >= cfg.duplicate_confirm_threshold and jac >= cfg.duplicate_jaccard_threshold
                canonical, later = documents[i], documents[j]
                canonical.duplicate_candidate = later.duplicate_candidate = True
                if confirmed and not later.is_duplicate and not canonical.is_duplicate:
                    later.is_duplicate = True
                    later.duplicate_of = canonical.document_id
                pairs.append(
                    DuplicatePair(
                        document_id=canonical.document_id,
                        other_document_id=later.document_id,
                        cosine_similarity=round(cosine, 4),
                        jaccard_similarity=round(jac, 4),
                        confirmed=confirmed,
                    )
                )
        return sorted(pairs, key=lambda p: -p.cosine_similarity)

    @staticmethod
    def _cluster_summaries(documents: list[Document]) -> list[ClusterInfo]:
        members: dict[int, list[Document]] = defaultdict(list)
        for doc in documents:
            if doc.cluster_id is not None and doc.cluster_id >= 0:
                members[doc.cluster_id].append(doc)
        summaries = []
        for cluster_id, docs in sorted(members.items()):
            categories = Counter(d.category for d in docs)
            summaries.append(
                ClusterInfo(
                    cluster_id=cluster_id,
                    size=len(docs),
                    document_ids=[d.document_id for d in docs],
                    titles=[d.title for d in docs],
                    dominant_category=categories.most_common(1)[0][0],
                    categories=dict(categories),
                )
            )
        return summaries
