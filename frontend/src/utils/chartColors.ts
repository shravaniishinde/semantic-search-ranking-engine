export const SYSTEM_COLORS: Record<string, string> = {
  bm25: 'var(--c-bm25)',
  semantic: 'var(--c-semantic)',
  hybrid: 'var(--c-hybrid)',
  'hybrid+rerank': 'var(--c-rerank)',
}

export const SYSTEM_LABELS: Record<string, string> = {
  bm25: 'BM25',
  semantic: 'Semantic (FAISS)',
  hybrid: 'Hybrid (RRF)',
  'hybrid+rerank': 'Hybrid + reranker',
}

/** Fixed colours for the sample corpus categories so every page agrees; others get a hashed colour. */
const KNOWN_CATEGORY_COLORS: Record<string, string> = {
  'cloud computing': '#0891b2',
  cybersecurity: '#d97706',
  databases: '#059669',
  'artificial intelligence': '#6366f1',
  'software engineering': '#64748b',
  networking: '#db2777',
  devops: '#7c3aed',
  miscellaneous: '#a3a3a3',
}
const FALLBACK_PALETTE = ['#65a30d', '#0ea5e9', '#e11d48', '#ca8a04', '#14b8a6', '#a855f7']

export function categoryColor(category: string): string {
  const known = KNOWN_CATEGORY_COLORS[category.toLowerCase()]
  if (known) return known
  let h = 0
  for (const ch of category) h = (h * 31 + ch.charCodeAt(0)) >>> 0
  return FALLBACK_PALETTE[h % FALLBACK_PALETTE.length]
}
