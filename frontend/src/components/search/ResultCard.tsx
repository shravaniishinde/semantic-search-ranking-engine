import { ChevronDown, ChevronUp, PanelRightOpen } from 'lucide-react'
import { useState } from 'react'
import { useApp } from '../../context/appState'
import type { SearchResultItem } from '../../types/api'
import { fmtNumber, humanize } from '../../utils/format'
import { EntityChips } from '../documents/EntityChips'
import { Badge } from '../ui/Badge'
import { RankingExplanation } from './RankingExplanation'

function Score({ label, value, color, digits = 3 }: { label: string; value: number | null; color: string; digits?: number }) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wide text-faint">{label}</div>
      <div className={`num text-[13px] font-semibold ${value === null ? 'text-faint' : color}`}>{value === null ? '—' : fmtNumber(value, digits)}</div>
    </div>
  )
}

export function ResultCard({ result }: { result: SearchResultItem }) {
  const [expanded, setExpanded] = useState(false)
  const { openDocument } = useApp()
  return (
    <article className="rounded-lg border border-line bg-surface" aria-label={`Result ${result.rank}: ${result.title}`}>
      <div className="flex gap-4 p-4">
        <div className="num w-8 shrink-0 pt-0.5 text-right text-lg font-semibold text-faint">#{result.rank}</div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-1">
            <button onClick={() => openDocument(result.document_id, result)} className="text-left text-[15px] font-semibold leading-snug text-fg hover:text-accent">
              {result.title}
            </button>
            <div className="flex items-center gap-1.5">
              <Badge>{humanize(result.category)}</Badge>
              <Badge>{result.source}</Badge>
            </div>
          </div>
          <div className="mt-0.5 font-mono text-[11px] text-faint">{result.document_id}</div>
          <p className="mt-2 line-clamp-2 text-[13px] leading-relaxed text-muted">{result.snippet}</p>
          <div className="mt-2">
            <EntityChips entities={result.entities} max={6} empty="" />
          </div>

          <div className="mt-3 flex flex-wrap items-end justify-between gap-3 border-t border-line pt-3">
            <div className="grid grid-cols-4 gap-x-6">
              <Score label="Final" value={result.final_score} color="text-rerank" />
              <Score label="BM25" value={result.bm25_score} color="text-bm25" />
              <Score label="Semantic" value={result.semantic_score} color="text-semantic" />
              <Score label="Hybrid" value={result.hybrid_score} color="text-hybrid" digits={4} />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => openDocument(result.document_id, result)}
                className="inline-flex items-center gap-1 rounded-md border border-line px-2.5 py-1 text-xs text-muted hover:bg-surface-2 hover:text-fg"
              >
                <PanelRightOpen size={13} /> Details
              </button>
              <button
                onClick={() => setExpanded((v) => !v)}
                aria-expanded={expanded}
                className="inline-flex items-center gap-1 rounded-md border border-accent/40 bg-accent-soft px-2.5 py-1 text-xs font-medium text-accent hover:opacity-90"
              >
                Why this result? {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              </button>
            </div>
          </div>
          {expanded && (
            <div className="mt-3">
              <RankingExplanation result={result} />
            </div>
          )}
        </div>
      </div>
    </article>
  )
}
