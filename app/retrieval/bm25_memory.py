"""In-memory BM25 (Okapi) retriever.

Used when Elasticsearch is not reachable (laptop without Docker, unit tests). It mirrors the ES
query used in ``ElasticsearchRetriever``: ``title^2 + content + entities`` summed per field with
the same BM25 defaults as Lucene (k1=1.2, b=0.75), so the two backends rank similarly. Scores
are not numerically identical because the analyzers differ.
"""

from __future__ import annotations

import math
from collections import Counter

from app.models.schemas import Document, RetrievalResult, SearchFilters
from app.preprocessing.text import tokenize
from app.retrieval.base import passes_filters

FIELD_BOOSTS = {"title": 2.0, "content": 1.0, "entities": 1.0}


class _FieldIndex:
    def __init__(self, k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        self.term_freqs: dict[str, Counter[str]] = {}
        self.lengths: dict[str, int] = {}
        self.doc_freq: Counter[str] = Counter()

    def add(self, doc_id: str, tokens: list[str]) -> None:
        self.remove(doc_id)
        tf = Counter(tokens)
        self.term_freqs[doc_id] = tf
        self.lengths[doc_id] = len(tokens)
        self.doc_freq.update(tf.keys())

    def remove(self, doc_id: str) -> None:
        tf = self.term_freqs.pop(doc_id, None)
        if tf is not None:
            self.lengths.pop(doc_id, None)
            self.doc_freq.subtract(tf.keys())

    def score(self, query_terms: list[str]) -> dict[str, float]:
        n_docs = len(self.term_freqs)
        if not n_docs:
            return {}
        avg_len = sum(self.lengths.values()) / n_docs or 1.0
        scores: dict[str, float] = {}
        for term in set(query_terms):
            df = self.doc_freq.get(term, 0)
            if df <= 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in self.term_freqs.items():
                freq = tf.get(term)
                if not freq:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / avg_len)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * freq * (self.k1 + 1) / (freq + norm)
        return scores


class InMemoryBM25Retriever:
    method = "bm25"
    backend = "memory"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.fields = {name: _FieldIndex(k1, b) for name in FIELD_BOOSTS}
        self.documents: dict[str, Document] = {}

    def count(self) -> int:
        return len(self.documents)

    def index_documents(self, documents: list[Document]) -> int:
        for doc in documents:
            self.documents[doc.document_id] = doc
            self.fields["title"].add(doc.document_id, tokenize(doc.title))
            self.fields["content"].add(doc.document_id, tokenize(doc.content))
            self.fields["entities"].add(doc.document_id, tokenize(" ".join(e.text for e in doc.entities)))
        return len(documents)

    def remove_documents(self, document_ids: list[str]) -> None:
        for doc_id in document_ids:
            self.documents.pop(doc_id, None)
            for field in self.fields.values():
                field.remove(doc_id)

    def retrieve(self, query: str, top_k: int, filters: SearchFilters | None = None) -> list[RetrievalResult]:
        terms = tokenize(query)
        if not terms:
            return []
        totals: dict[str, float] = {}
        for name, field in self.fields.items():
            for doc_id, score in field.score(terms).items():
                totals[doc_id] = totals.get(doc_id, 0.0) + FIELD_BOOSTS[name] * score
        ranked = sorted(
            ((d, s) for d, s in totals.items() if passes_filters(self.documents[d], filters)),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RetrievalResult(document_id=d, score=round(s, 6), rank=r, retrieval_method=self.method)
            for r, (d, s) in enumerate(ranked[:top_k], start=1)
        ]
