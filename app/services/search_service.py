"""SearchEngine: owns all components and orchestrates ingestion, training and search."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np

from app.classification.intent import IntentClassifier
from app.clustering.dbscan_analysis import DBSCANConfig, DocumentClusterAnalyzer
from app.config.settings import Settings
from app.embeddings.encoder import (
    Embedder,
    EmbeddingCache,
    SentenceTransformerEmbedder,
    embed_with_cache,
)
from app.ingestion.loader import load_documents
from app.models.schemas import (
    ClusterAnalysis,
    Document,
    DocumentIn,
    EmbeddingMetadata,
    HybridHit,
    IngestionReport,
    RejectedDocument,
    RetrievalDebug,
    RetrievalHit,
    RetrievalResult,
    SearchFilters,
    SearchResponse,
    SearchResultItem,
)
from app.nlp.ner import EntityExtractor
from app.preprocessing.text import clean_text, is_meaningful, tokenize
from app.ranking.features import QueryContext, RankingFeatureExtractor, to_matrix
from app.ranking.reranker import LTRDataset, XGBoostReranker
from app.retrieval.base import LexicalRetriever, passes_filters
from app.retrieval.bm25_memory import InMemoryBM25Retriever
from app.retrieval.elasticsearch_retriever import ElasticsearchRetriever
from app.retrieval.faiss_retriever import FaissIndex, FaissRetriever
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class QueryValidationError(ValueError):
    pass


def select_lexical_backend(settings: Settings) -> LexicalRetriever:
    """``memory`` -> in-memory BM25; ``elasticsearch`` -> ES (waits ~30s, then fails);
    ``auto`` -> ES if it answers a ping, else in-memory."""
    if settings.bm25_backend == "memory":
        return InMemoryBM25Retriever()
    es = ElasticsearchRetriever.from_url(settings.elasticsearch_url, settings.elasticsearch_index, settings.elasticsearch_timeout)
    attempts = 15 if settings.bm25_backend == "elasticsearch" else 1
    for attempt in range(attempts):
        if es.is_available():
            es.create_index()
            logger.info(
                "bm25.backend_selected", extra={"fields": {"backend": "elasticsearch", "url": settings.elasticsearch_url}}
            )
            return es
        if attempt + 1 < attempts:
            time.sleep(2)
    if settings.bm25_backend == "elasticsearch":
        raise RuntimeError(f"Elasticsearch not reachable at {settings.elasticsearch_url}")
    logger.warning("bm25.elasticsearch_unavailable_fallback_memory", extra={"fields": {"url": settings.elasticsearch_url}})
    return InMemoryBM25Retriever()


class SearchEngine:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        entity_extractor: EntityExtractor | None = None,
        lexical: LexicalRetriever | None = None,
        persist: bool = True,
    ) -> None:
        self.settings = settings
        self.persist = persist
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size)
        self.ner = entity_extractor or EntityExtractor(settings.spacy_model)
        self.lexical: LexicalRetriever = lexical if lexical is not None else select_lexical_backend(settings)
        self.cache = EmbeddingCache(settings.embedding_cache_path if persist else None)

        self.documents: dict[str, Document] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.faiss_index = FaissIndex(self.embedder.dimension)
        self.semantic = FaissRetriever(self.embedder, self.faiss_index, filter_fn=self._doc_passes)
        self.hybrid = HybridRetriever(self.lexical, self.semantic, rrf_k=settings.rrf_k, candidate_k=settings.candidate_k)
        self.intent = IntentClassifier(self.ner)
        self.features = RankingFeatureExtractor(candidate_depth=settings.candidate_k)
        self.reranker = XGBoostReranker()
        self.analyzer = DocumentClusterAnalyzer(
            DBSCANConfig(
                eps=settings.dbscan_eps,
                min_samples=settings.dbscan_min_samples,
                duplicate_candidate_threshold=settings.duplicate_candidate_threshold,
                duplicate_confirm_threshold=settings.duplicate_confirm_threshold,
                duplicate_jaccard_threshold=settings.duplicate_jaccard_threshold,
            )
        )
        self.analysis: ClusterAnalysis | None = None

        self._lock = threading.RLock()
        self._latencies: deque[float] = deque(maxlen=1000)
        self._search_count = 0
        self._zero_result_count = 0
        self._intent_counts: Counter[str] = Counter()
        self._candidate_total = 0

    # ================================================================== bootstrap
    @property
    def bm25_backend(self) -> str:
        return getattr(self.lexical, "backend", type(self.lexical).__name__)

    def bm25_available(self) -> bool:
        """Live reachability of the lexical backend (pings Elasticsearch; in-memory is always up)."""
        check = getattr(self.lexical, "is_available", None)
        return bool(check()) if callable(check) else True

    def bootstrap(self) -> None:
        """Load persisted state, or ingest the sample corpus; then make sure models exist."""
        loaded = self.load_state() if self.persist else False
        if not loaded and self.settings.load_sample_on_startup:
            docs, rejected = load_documents(self.settings.sample_documents_path)
            self.ingest(docs, pre_rejected=rejected)
        self.ensure_models(train_if_missing=self.settings.auto_train_models)

    def ensure_models(self, train_if_missing: bool = True) -> None:
        if not self.intent.is_trained and not self.intent.load(self.settings.intent_model_path) and train_if_missing:
            self.train_intent_classifier()
        if not self.reranker.is_trained and not self.reranker.load(self.settings.reranker_model_path) and train_if_missing:
            if self.documents:
                self.train_reranker()

    # ================================================================== ingestion
    def _doc_passes(self, document_id: str, filters: SearchFilters) -> bool:
        doc = self.documents.get(document_id)
        return doc is not None and passes_filters(doc, filters)

    def ingest(self, raw_docs: list[DocumentIn], pre_rejected: list[RejectedDocument] | None = None) -> IngestionReport:
        started = time.perf_counter()
        rejected = list(pre_rejected or [])
        accepted: dict[str, Document] = {}
        for raw in raw_docs:
            title, content = clean_text(raw.title), clean_text(raw.content)
            if not is_meaningful(content):
                rejected.append(RejectedDocument(document_id=raw.document_id, reason="empty or too short content after cleaning"))
                continue
            if raw.document_id in accepted:
                rejected.append(RejectedDocument(document_id=raw.document_id, reason="duplicate document_id in batch"))
                continue
            accepted[raw.document_id] = Document(
                document_id=raw.document_id,
                title=title,
                content=content,
                source=raw.source,
                category=raw.category.strip().lower(),
                created_at=raw.created_at,
                metadata=raw.metadata,
            )

        with self._lock:
            new_docs = list(accepted.values())
            if new_docs:
                for doc, ents in zip(new_docs, self.ner.extract_batch([d.full_text for d in new_docs])):
                    doc.entities = ents
                vectors, hashes, encoded = embed_with_cache(self.embedder, [d.full_text for d in new_docs], self.cache)
                for doc, vec, digest in zip(new_docs, vectors, hashes):
                    doc.embedding = EmbeddingMetadata(
                        model_name=self.embedder.model_name, dimension=int(vec.shape[0]), normalized=True, content_hash=digest
                    )
                    self.documents[doc.document_id] = doc
                    self.vectors[doc.document_id] = vec
                logger.info("ingestion.embedded", extra={"fields": {"documents": len(new_docs), "newly_encoded": encoded}})
            self._rebuild_indexes()
            if self.persist:
                self.save_state()

        report = IngestionReport(
            received=len(raw_docs) + len(pre_rejected or []),
            accepted=len(accepted),
            rejected=rejected,
            indexed=sum(d.indexed for d in self.documents.values()),
            excluded=sum(not d.indexed for d in self.documents.values()),
            duplicates=sum(d.is_duplicate for d in self.documents.values()),
            outliers=sum(d.is_outlier for d in self.documents.values()),
        )
        logger.info(
            "ingestion.completed",
            extra={
                "fields": {
                    **report.model_dump(exclude={"rejected"}),
                    "rejected": len(rejected),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            },
        )
        return report

    def _rebuild_indexes(self) -> None:
        """Re-run DBSCAN over the full corpus, decide what is indexable, rebuild FAISS, sync BM25."""
        docs = list(self.documents.values())
        if docs:
            self.analysis = self.analyzer.analyze(docs, np.stack([self.vectors[d.document_id] for d in docs]))
        for doc in docs:
            if doc.is_duplicate:
                doc.indexed, doc.exclusion_reason = False, f"confirmed duplicate of {doc.duplicate_of}"
            elif doc.is_outlier and self.settings.exclude_outliers_from_index:
                doc.indexed, doc.exclusion_reason = False, "DBSCAN outlier"
            else:
                doc.indexed, doc.exclusion_reason = True, None

        indexable = [d for d in docs if d.indexed]
        index = FaissIndex(self.embedder.dimension)
        if indexable:
            index.add([d.document_id for d in indexable], np.stack([self.vectors[d.document_id] for d in indexable]))
        self.faiss_index = index
        self.semantic.index = index

        excluded = [d.document_id for d in docs if not d.indexed]
        if excluded:
            self.lexical.remove_documents(excluded)
        self.lexical.index_documents(indexable)
        logger.info(
            "indexing.completed",
            extra={"fields": {"faiss": len(index), "bm25_backend": self.bm25_backend, "excluded": len(excluded)}},
        )

    # ================================================================== persistence
    def save_state(self) -> None:
        path = self.settings.document_store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [d.model_dump(mode="json") for d in self.documents.values()]
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        self.faiss_index.save(self.settings.faiss_index_path)
        self.cache.save()

    def load_state(self) -> bool:
        path = self.settings.document_store_path
        if not path.exists():
            return False
        docs = [Document.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]
        if not docs:
            return False
        with self._lock:
            vectors, _, encoded = embed_with_cache(self.embedder, [d.full_text for d in docs], self.cache)
            self.documents = {d.document_id: d for d in docs}
            self.vectors = {d.document_id: v for d, v in zip(docs, vectors)}
            self._rebuild_indexes()
        logger.info("state.loaded", extra={"fields": {"documents": len(docs), "re_encoded": encoded}})
        return True

    # ================================================================== training
    def train_intent_classifier(self, path: Path | None = None, save: bool = True) -> dict[str, Any]:
        queries, labels = IntentClassifier.load_training_data(path or self.settings.intent_training_path)
        self.intent.fit(queries, labels)
        if save and self.persist:
            self.intent.save(self.settings.intent_model_path)
        return {"examples": len(queries), "labels": dict(Counter(labels))}

    def query_context(self, query: str) -> QueryContext:
        return QueryContext(
            query=query,
            query_vector=self.embedder.encode_query(query),
            entities=self.ner.extract(query),
            intent=self.intent.predict(query),
        )

    def build_ltr_dataset(self, judged_queries: list[dict[str, Any]]) -> LTRDataset:
        features, labels, groups, queries = [], [], [], []
        for item in judged_queries:
            query, relevance = item["query"], item["relevance"]
            ctx = self.query_context(query)
            _, _, candidates = self.hybrid.retrieve_all(query, self.settings.candidate_k, query_vector=ctx.query_vector)
            if not candidates:
                continue
            rows = self.features.extract(ctx, candidates, self.documents, self.vectors)
            features.append(to_matrix(rows))
            labels.extend(relevance.get(c.document_id, 0) for c in candidates)
            groups.append(len(candidates))
            queries.append(query)
        matrix = np.vstack(features) if features else np.zeros((0, 0), dtype=np.float32)
        return LTRDataset(matrix, np.array(labels, dtype=np.float32), groups, queries)

    def train_reranker(self, qrels_path: Path | None = None, save: bool = True) -> dict[str, Any]:
        data = json.loads((qrels_path or self.settings.ranking_training_path).read_text(encoding="utf-8"))
        dataset = self.build_ltr_dataset(data["queries"])
        self.reranker.fit(dataset)
        if save and self.persist:
            self.reranker.save(self.settings.reranker_model_path)
        return {
            "queries": len(dataset.group_sizes),
            "rows": int(dataset.features.shape[0]),
            "feature_importance": self.reranker.feature_importance(),
        }

    # ================================================================== search
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
        include_retrieval: bool = True,
        rerank: bool = True,
        record_stats: bool = True,
    ) -> SearchResponse:
        started = time.perf_counter()
        stages: dict[str, float] = {}

        def lap(name: str, t0: float) -> float:
            now = time.perf_counter()
            stages[name] = round((now - t0) * 1000, 2)
            return now

        cleaned = clean_text(query)
        lap("preprocessing", started)
        if not cleaned or not (tokenize(cleaned, remove_stopwords=False)):
            raise QueryValidationError("query contains no searchable text")

        with self._lock:
            t = time.perf_counter()
            entities = self.ner.extract(cleaned)
            t = lap("ner", t)
            intent = self.intent.predict(cleaned)
            t = lap("intent", t)
            query_vector = self.embedder.encode_query(cleaned)
            t = lap("query_embedding", t)
            depth = max(top_k, self.settings.candidate_k)
            bm25, semantic, candidates = self.hybrid.retrieve_all(cleaned, depth, filters, query_vector=query_vector)
            t = lap("retrieval", t)
            stages.update(self.hybrid.last_timings_ms)  # bm25 / semantic / fusion breakdown of "retrieval"
            ctx = QueryContext(query=cleaned, query_vector=query_vector, entities=entities, intent=intent)
            rows = self.features.extract(ctx, candidates, self.documents, self.vectors)
            t = lap("features", t)
            if rerank:
                ranked = self.reranker.rerank(candidates, rows)
                ranker = self.reranker.name
            else:
                ranked = XGBoostReranker().rerank(candidates, rows)  # untrained -> hybrid order
                ranker = "hybrid"
            lap("rerank", t)

            results = [self._result_item(r) for r in ranked[:top_k]]
            retrieval = None
            if include_retrieval:
                retrieval = RetrievalDebug(
                    bm25=[self._hit(r) for r in bm25[:top_k]],
                    semantic=[self._hit(r) for r in semantic[:top_k]],
                    hybrid=[
                        HybridHit(
                            title=self.documents[c.document_id].title,
                            **c.model_dump(
                                include={
                                    "document_id",
                                    "hybrid_score",
                                    "bm25_score",
                                    "bm25_rank",
                                    "semantic_score",
                                    "semantic_rank",
                                }
                            ),
                        )
                        for c in candidates[:top_k]
                    ],
                    bm25_backend=self.bm25_backend
                    if self.hybrid.last_lexical_error is None
                    else f"{self.bm25_backend} (unavailable: {self.hybrid.last_lexical_error})",
                )

        latency = round((time.perf_counter() - started) * 1000, 2)
        if record_stats:
            self._record_search(latency, intent.label, len(candidates), len(results))
        logger.info(
            "search.completed",
            extra={
                "fields": {
                    "query": cleaned[:100] if self.settings.log_queries else "<redacted>",
                    "intent": intent.label,
                    "entities": len(entities),
                    "bm25_hits": len(bm25),
                    "semantic_hits": len(semantic),
                    "candidates": len(candidates),
                    "results": len(results),
                    "ranker": ranker,
                    "latency_ms": latency,
                }
            },
        )
        return SearchResponse(
            query=cleaned,
            intent=intent,
            entities=entities,
            retrieval=retrieval,
            results=results,
            ranker=ranker,
            candidate_count=len(candidates),
            retrieval_counts={"bm25": len(bm25), "semantic": len(semantic), "hybrid": len(candidates)},
            latency_ms=latency,
            stage_latency_ms=stages,
        )

    def _hit(self, r: RetrievalResult) -> RetrievalHit:
        return RetrievalHit(document_id=r.document_id, title=self.documents[r.document_id].title, score=r.score, rank=r.rank)

    def _result_item(self, r) -> SearchResultItem:
        doc = self.documents[r.document_id]
        snippet = doc.content if len(doc.content) <= 240 else doc.content[:240].rsplit(" ", 1)[0] + "..."
        return SearchResultItem(
            document_id=r.document_id,
            title=doc.title,
            snippet=snippet,
            category=doc.category,
            source=doc.source,
            rank=r.rank,
            final_score=r.final_score,
            bm25_score=r.bm25_score,
            semantic_score=r.semantic_score,
            hybrid_score=r.hybrid_score,
            bm25_rank=r.bm25_rank,
            semantic_rank=r.semantic_rank,
            features=r.features,
            entities=doc.entities,
        )

    def _record_search(self, latency: float, intent: str, candidates: int, results: int) -> None:
        self._search_count += 1
        self._latencies.append(latency)
        self._intent_counts[intent] += 1
        self._candidate_total += candidates
        if results == 0:
            self._zero_result_count += 1

    # ================================================================== read APIs
    def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    def list_documents(
        self,
        offset: int = 0,
        limit: int = 20,
        category: str | None = None,
        include_excluded: bool = True,
        q: str | None = None,
        source: str | None = None,
        cluster_id: int | None = None,
        is_duplicate: bool | None = None,
        duplicate_candidate: bool | None = None,
        is_outlier: bool | None = None,
    ) -> tuple[int, list[Document]]:
        docs = sorted(self.documents.values(), key=lambda d: d.document_id)
        if category:
            docs = [d for d in docs if d.category == category.lower()]
        if source:
            docs = [d for d in docs if d.source == source]
        if not include_excluded:
            docs = [d for d in docs if d.indexed]
        if cluster_id is not None:
            docs = [d for d in docs if d.cluster_id == cluster_id]
        if is_duplicate is not None:
            docs = [d for d in docs if d.is_duplicate == is_duplicate]
        if duplicate_candidate is not None:
            docs = [d for d in docs if d.duplicate_candidate == duplicate_candidate]
        if is_outlier is not None:
            docs = [d for d in docs if d.is_outlier == is_outlier]
        if q and q.strip():
            needle = q.strip().lower()
            docs = [d for d in docs if needle in d.document_id.lower() or needle in d.title.lower()]
        return len(docs), docs[offset : offset + limit]

    def analytics(self) -> dict[str, Any]:
        docs = list(self.documents.values())
        latencies = sorted(self._latencies)
        analysis = self.analysis
        return {
            "document_count": len(docs),
            "indexed_count": sum(d.indexed for d in docs),
            "excluded_count": sum(not d.indexed for d in docs),
            "faiss_vectors": len(self.faiss_index),
            "bm25_documents": self.lexical.count(),
            "bm25_backend": self.bm25_backend,
            "cluster_count": analysis.cluster_count if analysis else 0,
            "duplicate_candidate_count": analysis.duplicate_candidate_count if analysis else 0,
            "duplicate_count": analysis.confirmed_duplicate_count if analysis else 0,
            "outlier_count": analysis.outlier_count if analysis else 0,
            "categories": dict(Counter(d.category for d in docs)),
            "sources": dict(Counter(d.source for d in docs)),
            "entity_labels": dict(Counter(e.label for d in docs for e in d.entities)),
            "models": {
                "embedding_model": self.embedder.model_name,
                "ner_backend": self.ner.backend,
                "intent_classifier": "xgboost" if self.intent.is_trained else "rule_fallback",
                "reranker": self.reranker.name,
                # Gain-based importance from the trained XGBoost ranker ({} when untrained).
                "reranker_feature_importance": self.reranker.feature_importance(),
            },
            "search_stats": {
                "total_searches": self._search_count,
                "zero_result_searches": self._zero_result_count,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
                "p95_latency_ms": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
                "avg_candidates": round(self._candidate_total / self._search_count, 2) if self._search_count else None,
                "intent_distribution": dict(self._intent_counts),
            },
        }
