import type { SearchResponse } from '../../types/api'
import { fmtMs } from '../../utils/format'
import { Card } from '../ui/Card'

// Order and labels of the stage keys emitted by SearchEngine.search. "retrieval" is the parent of
// bm25/semantic/fusion, so it is not drawn separately when the breakdown exists.
const STAGES: { key: string; label: string; color: string }[] = [
  { key: 'preprocessing', label: 'Preprocessing', color: 'bg-line-strong' },
  { key: 'ner', label: 'NER', color: 'bg-line-strong' },
  { key: 'intent', label: 'Intent', color: 'bg-accent' },
  { key: 'query_embedding', label: 'Embedding', color: 'bg-semantic' },
  { key: 'bm25', label: 'BM25', color: 'bg-bm25' },
  { key: 'semantic', label: 'FAISS', color: 'bg-semantic' },
  { key: 'fusion', label: 'RRF fusion', color: 'bg-hybrid' },
  { key: 'features', label: 'Features', color: 'bg-line-strong' },
  { key: 'rerank', label: 'Reranking', color: 'bg-rerank' },
]

export function LatencyBreakdown({ response }: { response: SearchResponse }) {
  const s = response.stage_latency_ms
  const hasBreakdown = 'bm25' in s
  const rows = STAGES.filter((st) => st.key in s).map((st) => ({ ...st, ms: s[st.key] }))
  if (!hasBreakdown && 'retrieval' in s) rows.splice(4, 0, { key: 'retrieval', label: 'Retrieval', color: 'bg-hybrid', ms: s.retrieval })
  const measured = rows.reduce((sum, r) => sum + r.ms, 0)
  const other = Math.max(0, response.latency_ms - measured)
  const max = Math.max(...rows.map((r) => r.ms), other, 0.001)

  return (
    <Card title="Search performance" subtitle="Server-side stage timings for this request" actions={<span className="num text-sm font-semibold">{fmtMs(response.latency_ms)}</span>}>
      <div className="space-y-1.5" role="list" aria-label="Stage latencies">
        {rows.map((r) => (
          <div key={r.key} role="listitem" className="grid grid-cols-[88px_1fr_64px] items-center gap-2 text-xs">
            <span className="text-muted">{r.label}</span>
            <div className="h-2 rounded-sm bg-surface-2">
              <div className={`h-2 rounded-sm ${r.color}`} style={{ width: `${Math.max(1, (r.ms / max) * 100)}%` }} />
            </div>
            <span className="num text-right text-fg">{fmtMs(r.ms)}</span>
          </div>
        ))}
        <div className="grid grid-cols-[88px_1fr_64px] items-center gap-2 text-xs" title="Total minus the measured stages: response assembly, locking, logging">
          <span className="text-faint">Other</span>
          <div className="h-2 rounded-sm bg-surface-2">
            <div className="h-2 rounded-sm bg-line" style={{ width: `${Math.max(1, (other / max) * 100)}%` }} />
          </div>
          <span className="num text-right text-faint">{fmtMs(other)}</span>
        </div>
      </div>
    </Card>
  )
}
