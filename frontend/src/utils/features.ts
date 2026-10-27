// Presentation metadata for the reranker's feature vector (app/ranking/features.py FEATURE_NAMES).
// This only describes how to *display* each feature; every value comes from the API.
import type { RankingFeatures } from '../types/api'
import { NA, fmtNumber, fmtPercent, humanize } from './format'

export type FeatureKind = 'score' | 'fraction' | 'rank' | 'flag' | 'count' | 'logLength'

export interface FeatureMeta {
  label: string
  kind: FeatureKind
  group: 'Retrieval signals' | 'Query–document match' | 'Query & document'
  help: string
}

export const FEATURE_META: Record<string, FeatureMeta> = {
  bm25_score: { label: 'BM25 score', kind: 'score', group: 'Retrieval signals', help: 'Raw lexical score (0 if BM25 did not return the document).' },
  bm25_score_norm: { label: 'BM25 score (normalised)', kind: 'fraction', group: 'Retrieval signals', help: 'BM25 score divided by the best BM25 score among this query’s candidates.' },
  bm25_rank: { label: 'BM25 rank', kind: 'rank', group: 'Retrieval signals', help: 'Position in the BM25 list.' },
  in_bm25: { label: 'Found by BM25', kind: 'flag', group: 'Retrieval signals', help: 'Whether the lexical retriever returned this document.' },
  semantic_similarity: { label: 'Semantic similarity', kind: 'fraction', group: 'Retrieval signals', help: 'Exact cosine similarity between query and document embeddings.' },
  semantic_rank: { label: 'Semantic rank', kind: 'rank', group: 'Retrieval signals', help: 'Position in the FAISS list.' },
  in_semantic: { label: 'Found by FAISS', kind: 'flag', group: 'Retrieval signals', help: 'Whether the dense retriever returned this document.' },
  hybrid_score: { label: 'Hybrid (RRF) score', kind: 'score', group: 'Retrieval signals', help: 'Reciprocal Rank Fusion of both ranks.' },
  hybrid_rank: { label: 'Hybrid rank', kind: 'rank', group: 'Retrieval signals', help: 'Position after fusion, before reranking.' },
  query_term_coverage: { label: 'Query term coverage', kind: 'fraction', group: 'Query–document match', help: 'Share of query tokens that occur in the document.' },
  title_term_coverage: { label: 'Title term coverage', kind: 'fraction', group: 'Query–document match', help: 'Share of query tokens that occur in the title.' },
  title_phrase_match: { label: 'Exact title phrase', kind: 'flag', group: 'Query–document match', help: 'Whole query appears verbatim in the title.' },
  entity_overlap: { label: 'Entity overlap', kind: 'fraction', group: 'Query–document match', help: 'Share of query entities found in the document.' },
  category_match: { label: 'Category match', kind: 'flag', group: 'Query–document match', help: 'Query mentions vocabulary of the document’s category.' },
  doc_length_log: { label: 'Document length', kind: 'logLength', group: 'Query & document', help: 'log(1 + word count) of the document content.' },
  query_length: { label: 'Query length', kind: 'count', group: 'Query & document', help: 'Number of words in the query.' },
  intent_confidence: { label: 'Intent confidence', kind: 'fraction', group: 'Query & document', help: 'Classifier confidence for the predicted intent.' },
}

const INTENT_PREFIX = 'intent_'

export interface FeatureRow {
  key: string
  label: string
  group: FeatureMeta['group']
  help: string
  display: string
  /** 0..1 for a bar, or null when a bar would be meaningless. */
  bar: number | null
}

/**
 * Turns the raw feature dict into display rows. Intent one-hot features are folded into one
 * "Predicted intent" row. A rank of `candidate_k + 1` is how the backend encodes "not retrieved".
 */
export function featureRows(features: RankingFeatures): FeatureRow[] {
  const rows: FeatureRow[] = []
  for (const [key, value] of Object.entries(features)) {
    if (key.startsWith(INTENT_PREFIX) && key !== 'intent_confidence') continue
    const meta = FEATURE_META[key] ?? { label: humanize(key), kind: 'score', group: 'Query & document', help: '' }
    let display: string
    let bar: number | null = null
    switch (meta.kind) {
      case 'fraction':
        display = key === 'semantic_similarity' ? fmtNumber(value) : fmtPercent(value, 0)
        bar = Math.max(0, Math.min(1, value))
        break
      case 'flag':
        display = value >= 0.5 ? 'Yes' : 'No'
        break
      case 'rank': {
        const presence = key === 'bm25_rank' ? features.in_bm25 : key === 'semantic_rank' ? features.in_semantic : 1
        display = presence === 0 ? 'not retrieved' : `#${Math.round(value)}`
        break
      }
      case 'count':
        display = String(Math.round(value))
        break
      case 'logLength':
        // The backend sends log1p(word count); invert it to show the word count.
        display = `${Math.round(Math.expm1(value))} words`
        break
      default:
        display = fmtNumber(value, key === 'hybrid_score' ? 4 : 3)
    }
    rows.push({ key, label: meta.label, group: meta.group, help: meta.help, display, bar })
  }
  const activeIntent = Object.entries(features).find(([k, v]) => k.startsWith(INTENT_PREFIX) && k !== 'intent_confidence' && v === 1)
  rows.push({
    key: 'intent_onehot',
    label: 'Intent (one-hot)',
    group: 'Query & document',
    help: 'Intent is passed to the reranker as one-hot features.',
    display: activeIntent ? humanize(activeIntent[0].slice(INTENT_PREFIX.length)) : NA,
    bar: null,
  })
  return rows
}

export const FEATURE_GROUPS: FeatureMeta['group'][] = ['Retrieval signals', 'Query–document match', 'Query & document']

export function featureLabel(key: string): string {
  return FEATURE_META[key]?.label ?? humanize(key)
}
