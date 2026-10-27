"""Pydantic models shared by ingestion, the search pipeline and the API."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


# --------------------------------------------------------------------------- documents
class Entity(BaseModel):
    text: str
    label: str
    start: int | None = None
    end: int | None = None


class EmbeddingMetadata(BaseModel):
    model_name: str
    dimension: int
    normalized: bool = True
    content_hash: str


class DocumentIn(BaseModel):
    """A raw document as accepted by ingestion (JSON / JSONL / API)."""

    model_config = ConfigDict(populate_by_name=True)

    document_id: str | None = Field(default=None, validation_alias=AliasChoices("document_id", "id"))
    title: str = ""
    content: str = Field(default="", validation_alias=AliasChoices("content", "text", "body"))
    source: str = "unknown"
    category: str = "uncategorized"
    created_at: date | None = Field(default=None, validation_alias=AliasChoices("created_at", "date"))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_id(self) -> DocumentIn:
        if not self.document_id:
            digest = hashlib.sha1(f"{self.title}\n{self.content}".encode()).hexdigest()[:12]
            self.document_id = f"doc-{digest}"
        return self


class Document(BaseModel):
    """A processed document as stored by the engine."""

    document_id: str
    title: str
    content: str
    source: str = "unknown"
    category: str = "uncategorized"
    created_at: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    entities: list[Entity] = Field(default_factory=list)
    embedding: EmbeddingMetadata | None = None

    # DBSCAN analysis
    cluster_id: int | None = None  # -1 = DBSCAN noise
    duplicate_candidate: bool = False  # high embedding similarity to another doc
    is_duplicate: bool = False  # confirmed duplicate (embedding + lexical check)
    duplicate_of: str | None = None
    is_outlier: bool = False

    indexed: bool = False
    exclusion_reason: str | None = None

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.content}" if self.title else self.content


# --------------------------------------------------------------------------- retrieval
class RetrievalResult(BaseModel):
    document_id: str
    score: float
    rank: int
    retrieval_method: str


class HybridCandidate(BaseModel):
    document_id: str
    hybrid_score: float
    bm25_score: float | None = None
    bm25_rank: int | None = None
    semantic_score: float | None = None
    semantic_rank: int | None = None
    retrieval_method: str = "hybrid"
    sources: list[str] = Field(default_factory=list)


class RankedResult(BaseModel):
    document_id: str
    rank: int
    final_score: float
    bm25_score: float | None
    semantic_score: float | None
    hybrid_score: float
    bm25_rank: int | None
    semantic_rank: int | None
    features: dict[str, float]
    ranker: str


class IntentPrediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float] = Field(default_factory=dict)
    source: str  # "xgboost" or "rule_fallback"


# --------------------------------------------------------------------------- API contracts
class SearchFilters(BaseModel):
    category: str | None = None
    source: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: SearchFilters | None = None
    include_retrieval: bool = True

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class RetrievalHit(BaseModel):
    document_id: str
    title: str
    score: float
    rank: int


class HybridHit(BaseModel):
    document_id: str
    title: str
    hybrid_score: float
    bm25_score: float | None
    bm25_rank: int | None
    semantic_score: float | None
    semantic_rank: int | None


class RetrievalDebug(BaseModel):
    bm25: list[RetrievalHit]
    semantic: list[RetrievalHit]
    hybrid: list[HybridHit]
    bm25_backend: str


class SearchResultItem(BaseModel):
    document_id: str
    title: str
    snippet: str
    category: str
    source: str
    rank: int
    final_score: float
    bm25_score: float | None
    semantic_score: float | None
    hybrid_score: float
    bm25_rank: int | None
    semantic_rank: int | None
    features: dict[str, float]
    entities: list[Entity]


class SearchResponse(BaseModel):
    query: str
    intent: IntentPrediction
    entities: list[Entity]
    retrieval: RetrievalDebug | None
    results: list[SearchResultItem]
    ranker: str
    candidate_count: int
    # Full candidate-list sizes before truncation to top_k: {"bm25", "semantic", "hybrid"}.
    retrieval_counts: dict[str, int] = Field(default_factory=dict)
    latency_ms: float
    stage_latency_ms: dict[str, float]


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)


class RejectedDocument(BaseModel):
    document_id: str | None
    reason: str


class IngestionReport(BaseModel):
    received: int
    accepted: int
    rejected: list[RejectedDocument]
    indexed: int
    excluded: int
    duplicates: int
    outliers: int


class DocumentListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    documents: list[Document]


class ClusterInfo(BaseModel):
    cluster_id: int
    size: int
    document_ids: list[str]
    titles: list[str]
    dominant_category: str | None
    categories: dict[str, int]


class DuplicatePair(BaseModel):
    document_id: str
    other_document_id: str
    cosine_similarity: float
    jaccard_similarity: float
    confirmed: bool


class ClusterAnalysis(BaseModel):
    total_documents: int
    cluster_count: int
    clustered_documents: int
    outlier_count: int
    duplicate_candidate_count: int
    confirmed_duplicate_count: int
    eps: float
    min_samples: int
    clusters: list[ClusterInfo]
    outliers: list[str]
    duplicate_pairs: list[DuplicatePair]
