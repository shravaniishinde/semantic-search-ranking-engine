// TypeScript mirrors of the FastAPI response models in app/models/schemas.py and the
// dict-shaped responses built in app/api/main.py / app/services/search_service.py.

export interface Entity {
  text: string
  label: string
  start: number | null
  end: number | null
}

export interface EmbeddingMetadata {
  model_name: string
  dimension: number
  normalized: boolean
  content_hash: string
}

export interface Document {
  document_id: string
  title: string
  content: string
  source: string
  category: string
  created_at: string | null
  metadata: Record<string, unknown>
  entities: Entity[]
  embedding: EmbeddingMetadata | null
  cluster_id: number | null
  duplicate_candidate: boolean
  is_duplicate: boolean
  duplicate_of: string | null
  is_outlier: boolean
  indexed: boolean
  exclusion_reason: string | null
}

export interface DocumentListResponse {
  total: number
  offset: number
  limit: number
  documents: Document[]
}

export interface DocumentQuery {
  offset?: number
  limit?: number
  q?: string
  category?: string
  source?: string
  cluster_id?: number
  include_excluded?: boolean
  is_duplicate?: boolean
  duplicate_candidate?: boolean
  is_outlier?: boolean
}

// ------------------------------------------------------------------ search
export type IntentLabel = 'informational' | 'technical' | 'entity_lookup' | 'troubleshooting' | 'comparison' | string

export interface IntentPrediction {
  label: IntentLabel
  confidence: number
  probabilities: Record<string, number>
  source: 'xgboost' | 'rule_fallback' | string
}

export interface SearchFilters {
  category?: string | null
  source?: string | null
}

export interface SearchRequest {
  query: string
  top_k?: number
  filters?: SearchFilters | null
  include_retrieval?: boolean
}

export interface RetrievalHit {
  document_id: string
  title: string
  score: number
  rank: number
}

export interface HybridHit {
  document_id: string
  title: string
  hybrid_score: number
  bm25_score: number | null
  bm25_rank: number | null
  semantic_score: number | null
  semantic_rank: number | null
}

export interface RetrievalDebug {
  bm25: RetrievalHit[]
  semantic: RetrievalHit[]
  hybrid: HybridHit[]
  bm25_backend: string
}

/** Ranking features computed by app/ranking/features.py (keys of FEATURE_NAMES). */
export type RankingFeatures = Record<string, number>

export interface SearchResultItem {
  document_id: string
  title: string
  snippet: string
  category: string
  source: string
  rank: number
  final_score: number
  bm25_score: number | null
  semantic_score: number | null
  hybrid_score: number
  bm25_rank: number | null
  semantic_rank: number | null
  features: RankingFeatures
  entities: Entity[]
}

export interface SearchResponse {
  query: string
  intent: IntentPrediction
  entities: Entity[]
  retrieval: RetrievalDebug | null
  results: SearchResultItem[]
  ranker: string
  candidate_count: number
  retrieval_counts: Partial<Record<'bm25' | 'semantic' | 'hybrid', number>>
  latency_ms: number
  stage_latency_ms: Record<string, number>
}

// ------------------------------------------------------------------ ingestion
export interface DocumentIn {
  document_id?: string
  title?: string
  content?: string
  text?: string
  source?: string
  category?: string
  created_at?: string
  metadata?: Record<string, unknown>
  [key: string]: unknown
}

export interface RejectedDocument {
  document_id: string | null
  reason: string
}

export interface IngestionReport {
  received: number
  accepted: number
  rejected: RejectedDocument[]
  /** The following four are corpus-wide totals after the ingest, not per-batch deltas. */
  indexed: number
  excluded: number
  duplicates: number
  outliers: number
}

// ------------------------------------------------------------------ clusters
export interface ClusterInfo {
  cluster_id: number
  size: number
  document_ids: string[]
  titles: string[]
  dominant_category: string | null
  categories: Record<string, number>
}

export interface DuplicatePair {
  document_id: string
  other_document_id: string
  cosine_similarity: number
  jaccard_similarity: number
  confirmed: boolean
}

export interface ClusterAnalysis {
  total_documents: number
  cluster_count: number
  clustered_documents: number
  outlier_count: number
  duplicate_candidate_count: number
  confirmed_duplicate_count: number
  eps: number
  min_samples: number
  clusters: ClusterInfo[]
  outliers: string[]
  duplicate_pairs: DuplicatePair[]
}

// ------------------------------------------------------------------ analytics / health / evaluation
export interface SearchStats {
  total_searches: number
  zero_result_searches: number
  avg_latency_ms: number | null
  p95_latency_ms: number | null
  avg_candidates: number | null
  intent_distribution: Record<string, number>
}

export interface Analytics {
  document_count: number
  indexed_count: number
  excluded_count: number
  faiss_vectors: number
  bm25_documents: number
  bm25_backend: string
  cluster_count: number
  duplicate_candidate_count: number
  duplicate_count: number
  outlier_count: number
  categories: Record<string, number>
  sources: Record<string, number>
  entity_labels: Record<string, number>
  models: {
    embedding_model: string
    ner_backend: string
    intent_classifier: string
    reranker: string
    reranker_feature_importance: Record<string, number>
  }
  search_stats: SearchStats
}

export interface Health {
  status: string
  documents: number
  indexed: number
  bm25_backend: string
  bm25_backend_requested: 'auto' | 'elasticsearch' | 'memory' | string
  /** Live reachability of the active BM25 backend (absent on older backends). */
  bm25_available?: boolean
  bm25_documents: number
  faiss_vectors: number
  embedding_model: string
  ner_backend: string
  intent_model: string
  reranker: string
}

export type SystemName = 'bm25' | 'semantic' | 'hybrid' | 'hybrid+rerank'

export interface EvaluationReport {
  generated_at?: string
  ks?: number[]
  queries: number
  ranker: string
  bm25_backend: string
  metrics: Record<SystemName, Record<string, number>>
  per_query_first_relevant_rank: Record<SystemName, Record<string, number | null>>
}

export interface EvaluationResponse {
  available: boolean
  report: EvaluationReport | null
}
