import type { SearchResultItem } from '../../types/api'
import { FEATURE_GROUPS, featureRows } from '../../utils/features'
import { fmtNumber } from '../../utils/format'

/**
 * Shows exactly the feature vector the reranker scored. XGBoost combines these non-linearly and
 * the backend does not compute per-feature attributions, so no contribution numbers are shown.
 */
export function RankingExplanation({ result }: { result: SearchResultItem }) {
  const rows = featureRows(result.features)
  const hybridRank = result.features.hybrid_rank
  const movement = hybridRank ? Math.round(hybridRank) - result.rank : 0
  return (
    <div className="@container rounded-md border border-line bg-bg/60 p-3" data-testid="ranking-explanation">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-[13px] font-semibold">Ranking explanation</h3>
        <p className="num text-xs text-muted">
          Hybrid #{hybridRank ? Math.round(hybridRank) : '?'} → final #{result.rank}
          {movement !== 0 && <span className={movement > 0 ? 'text-ok' : 'text-bad'}> ({movement > 0 ? `↑${movement}` : `↓${-movement}`})</span>}
          {' · '}reranker score <span className="text-rerank">{fmtNumber(result.final_score)}</span>
        </p>
      </div>
      <div className="grid gap-4 @xl:grid-cols-2 @4xl:grid-cols-3">
        {FEATURE_GROUPS.map((group) => (
          <div key={group}>
            <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-faint">{group}</div>
            <table className="w-full text-xs">
              <tbody>
                {rows
                  .filter((r) => r.group === group)
                  .map((r) => (
                    <tr key={r.key} title={r.help} className="border-b border-line/60 last:border-0">
                      <td className="py-1 pr-2 text-muted">{r.label}</td>
                      <td className="py-1 text-right">
                        <div className="flex items-center justify-end gap-2">
                          {r.bar !== null && (
                            <div className="h-1 w-12 rounded-full bg-surface-2">
                              <div className="h-1 rounded-full bg-accent" style={{ width: `${r.bar * 100}%` }} />
                            </div>
                          )}
                          <span className="num font-medium text-fg">{r.display}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
      <p className="mt-3 text-[11px] leading-relaxed text-faint">
        These are the inputs the XGBoost LambdaMART-style ranker scored for this document. The model combines them
        non-linearly; see Analytics for the model&apos;s global feature importance. Hover a row for its definition.
      </p>
    </div>
  )
}
