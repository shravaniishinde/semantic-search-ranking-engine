import type { Health } from '../types/api'

export type Tone = 'ok' | 'warn' | 'bad' | 'info' | 'neutral'

export interface Bm25Status {
  tone: Tone
  status: string
  /** Banner shown above the service list; null when nothing needs explaining. */
  banner: { title: string; message: string; tone: 'info' | 'warn' } | null
}

/**
 * "memory" can be deliberate (BM25_BACKEND=memory) or an automatic fallback (auto + ES
 * unreachable at startup). "elasticsearch" with bm25_available=false means ES stopped answering
 * after startup: searches continue semantic-only until it is back.
 */
export function describeBm25(h: Health): Bm25Status {
  if (h.bm25_backend === 'elasticsearch') {
    if (h.bm25_available === false)
      return {
        tone: 'warn',
        status: 'Elasticsearch unreachable',
        banner: {
          tone: 'warn',
          title: 'Retrieval backend: Elasticsearch unreachable',
          message:
            'Elasticsearch stopped responding. Searches continue with semantic (FAISS) retrieval only; BM25 results return when Elasticsearch is back.',
        },
      }
    return { tone: 'ok', status: 'Elasticsearch', banner: null }
  }
  if (h.bm25_backend === 'memory' && h.bm25_backend_requested === 'memory')
    return { tone: 'ok', status: 'In-memory BM25 (configured)', banner: null }
  if (h.bm25_backend === 'memory')
    return {
      tone: 'info',
      status: 'In-memory fallback',
      banner: {
        tone: 'info',
        title: 'Retrieval backend: in-memory fallback',
        message:
          'Elasticsearch is unavailable. Search is using the local BM25 fallback (same fields, boosts and BM25 parameters). Retrieval works normally.',
      },
    }
  return { tone: 'warn', status: h.bm25_backend, banner: null }
}
