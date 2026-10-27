import type { SearchResponse } from '../../types/api'
import { fmtMs, fmtPercent, humanize } from '../../utils/format'
import { EntityChips } from '../documents/EntityChips'
import { Badge } from '../ui/Badge'
import { SectionLabel } from '../ui/Card'

export function QueryInsights({ response }: { response: SearchResponse }) {
  const { intent } = response
  const probabilities = Object.entries(intent.probabilities).sort((a, b) => b[1] - a[1])
  return (
    <section aria-label="Query insights" className="grid gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-[1.2fr_1.5fr_0.8fr]">
      <div className="bg-surface p-4">
        <SectionLabel>Predicted intent</SectionLabel>
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="text-lg font-semibold" data-testid="intent-label">
            {humanize(intent.label)}
          </span>
          <span className="num text-sm text-muted" data-testid="intent-confidence">
            {intent.source === 'rule_fallback' ? 'confidence N/A' : `${fmtPercent(intent.confidence)} confidence`}
          </span>
        </div>
        <div className="mt-1 text-xs text-faint">{intent.source === 'xgboost' ? 'XGBoost classifier' : 'Keyword rule fallback (no trained model)'}</div>
        {probabilities.length > 0 && (
          <div className="mt-3 space-y-1">
            {probabilities.map(([label, p]) => (
              <div key={label} className="flex items-center gap-2 text-xs">
                <span className="w-28 shrink-0 truncate text-muted">{humanize(label)}</span>
                <div className="h-1.5 flex-1 rounded-full bg-surface-2">
                  <div className={`h-1.5 rounded-full ${label === intent.label ? 'bg-accent' : 'bg-line-strong'}`} style={{ width: `${p * 100}%` }} />
                </div>
                <span className="num w-11 text-right text-faint">{fmtPercent(p, 1)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="bg-surface p-4">
        <SectionLabel>Extracted entities (spaCy NER)</SectionLabel>
        <EntityChips entities={response.entities} empty="No entities found in the query" />
        <div className="mt-4">
          <SectionLabel>Normalised query</SectionLabel>
          <code className="break-words font-mono text-[13px] text-fg">{response.query}</code>
        </div>
      </div>
      <div className="bg-surface p-4">
        <SectionLabel>Server latency</SectionLabel>
        <div className="num text-2xl font-semibold" data-testid="total-latency">
          {fmtMs(response.latency_ms)}
        </div>
        <div className="mt-3 space-y-1.5 text-xs text-muted">
          <div className="flex justify-between gap-2">
            <span>Ranker</span>
            <Badge tone={response.ranker === 'xgboost_ltr' ? 'rerank' : 'warn'}>{response.ranker}</Badge>
          </div>
          <div className="flex justify-between gap-2">
            <span>Candidates</span>
            <span className="num text-fg">{response.candidate_count}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span>Returned</span>
            <span className="num text-fg">{response.results.length}</span>
          </div>
        </div>
      </div>
    </section>
  )
}
