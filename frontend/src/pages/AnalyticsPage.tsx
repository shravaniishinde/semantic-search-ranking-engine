import { Boxes, Copy, Database, FileText, Loader2, Play, ScanSearch } from 'lucide-react'
import { GroupedBarChart, HBarChart } from '../components/charts/Charts'
import { SYSTEM_COLORS, SYSTEM_LABELS, categoryColor } from '../utils/chartColors'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { StatCard, StatGrid } from '../components/ui/StatCard'
import { EmptyState, ErrorState, LoadingState } from '../components/ui/States'
import { useAnalytics, useClusters, useEvaluation, useRunEvaluation } from '../hooks/queries'
import type { EvaluationReport, SystemName } from '../types/api'
import { featureLabel } from '../utils/features'
import { fmtDateTime, fmtInt, fmtMs, fmtNumber, humanize } from '../utils/format'

const SYSTEMS: SystemName[] = ['bm25', 'semantic', 'hybrid', 'hybrid+rerank']

function EvaluationSection() {
  const evaluation = useEvaluation()
  const run = useRunEvaluation()
  const report = evaluation.data?.report ?? null
  const metricNames = report ? Object.keys(report.metrics[SYSTEMS[0]] ?? {}) : []

  const runButton = (
    <Button onClick={() => run.mutate()} disabled={run.isPending}>
      {run.isPending ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
      {report ? 'Re-run evaluation' : 'Run evaluation'}
    </Button>
  )

  return (
    <Card
      title="Retrieval quality"
      subtitle={
        report
          ? `${report.queries} held-out queries · graded relevance judgments · generated ${fmtDateTime(report.generated_at)} · BM25 backend: ${report.bm25_backend}`
          : 'Offline evaluation on held-out judged queries'
      }
      actions={runButton}
    >
      {evaluation.isLoading && <LoadingState />}
      {evaluation.error && <ErrorState error={evaluation.error} onRetry={() => evaluation.refetch()} />}
      {run.error && <ErrorState error={run.error} compact />}
      {evaluation.data && !report && (
        <EmptyState
          title="No evaluation report yet"
          description="Run the held-out evaluation (≈10 s) or `python scripts/evaluate.py`. Metrics are computed by the backend; nothing is shown until they exist."
        />
      )}
      {report && metricNames.length > 0 && <EvaluationBody report={report} metricNames={metricNames} />}
    </Card>
  )
}

function EvaluationBody({ report, metricNames }: { report: EvaluationReport; metricNames: string[] }) {
  const chartData = metricNames.map((m) => ({ name: m, ...Object.fromEntries(SYSTEMS.map((s) => [s, report.metrics[s]?.[m] ?? 0])) }))
  const best = (m: string) => Math.max(...SYSTEMS.map((s) => report.metrics[s]?.[m] ?? -1))
  return (
    <div className="grid gap-6 xl:grid-cols-[1.1fr_1fr]">
      <GroupedBarChart
        ariaLabel="Evaluation metrics by retrieval system"
        data={chartData}
        domain={[0, 1]}
        height={280}
        series={SYSTEMS.map((s) => ({ key: s, label: SYSTEM_LABELS[s], color: SYSTEM_COLORS[s] }))}
      />
      <div className="min-w-0">
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]" aria-label="Evaluation metrics">
            <thead className="border-b border-line text-[11px] uppercase tracking-wide text-faint">
              <tr>
                <th className="py-2 pr-2 text-left font-semibold">System</th>
                {metricNames.map((m) => (
                  <th key={m} className="px-1.5 py-2 text-right font-semibold">
                    {m}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {SYSTEMS.map((s) => (
                <tr key={s} className="border-b border-line/60 last:border-0">
                  <td className="whitespace-nowrap py-2 pr-2">
                    <span className="mr-1.5 inline-block h-2 w-2 rounded-sm" style={{ background: SYSTEM_COLORS[s] }} />
                    {SYSTEM_LABELS[s]}
                  </td>
                  {metricNames.map((m) => {
                    const v = report.metrics[s]?.[m]
                    return (
                      <td key={m} className={`num px-1.5 py-2 text-right ${v === best(m) ? 'font-semibold text-fg' : 'text-muted'}`}>
                        {fmtNumber(v)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[11px] leading-relaxed text-faint">
          Best value per column in bold. Small hand-labelled dataset ({report.queries} queries): differences of a few points are not statistically
          meaningful. P@K is low by construction because most queries have only 1–3 relevant documents.
        </p>
      </div>
    </div>
  )
}

export function AnalyticsPage() {
  const analytics = useAnalytics()
  const clusters = useClusters()

  if (analytics.isLoading) return <LoadingState label="Loading analytics…" />
  if (analytics.error) return <ErrorState error={analytics.error} onRetry={() => analytics.refetch()} />
  const a = analytics.data
  if (!a) return null

  const categoryData = Object.entries(a.categories)
    .sort((x, y) => y[1] - x[1])
    .map(([name, value]) => ({ name: humanize(name), value, color: categoryColor(name) }))
  const clusterData = (clusters.data?.clusters ?? []).map((c) => ({ name: `#${c.cluster_id}`, value: c.size }))
  const importance = Object.entries(a.models.reranker_feature_importance ?? {})
    .filter(([, v]) => v > 0)
    .slice(0, 12)
    .map(([k, v]) => ({ name: featureLabel(k), value: v }))
  const intentData = Object.entries(a.search_stats.intent_distribution).map(([k, v]) => ({ name: humanize(k), value: v }))
  const entityData = Object.entries(a.entity_labels)
    .sort((x, y) => y[1] - x[1])
    .map(([name, value]) => ({ name, value }))
  const s = a.search_stats

  return (
    <div className="space-y-4">
      <StatGrid>
        <StatCard label="Documents" value={fmtInt(a.document_count)} icon={FileText} />
        <StatCard label="Indexed" value={fmtInt(a.indexed_count)} hint={`FAISS ${a.faiss_vectors} · BM25 ${a.bm25_documents}`} icon={Database} />
        <StatCard label="Clusters" value={fmtInt(a.cluster_count)} icon={Boxes} />
        <StatCard label="Duplicates" value={fmtInt(a.duplicate_count)} hint={`${a.duplicate_candidate_count} candidates`} icon={Copy} />
        <StatCard label="Outliers" value={fmtInt(a.outlier_count)} icon={ScanSearch} />
      </StatGrid>

      <EvaluationSection />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Documents by category">
          {categoryData.length ? <HBarChart ariaLabel="Documents by category" data={categoryData} labelWidth={160} /> : <EmptyState title="No documents" />}
        </Card>
        <Card title="Documents per cluster" subtitle="DBSCAN clusters (noise points excluded)">
          {clusters.isLoading ? <LoadingState /> : clusterData.length ? <HBarChart ariaLabel="Documents per cluster" data={clusterData} labelWidth={40} color="var(--accent)" /> : <EmptyState title="No clusters" />}
        </Card>
        <Card title="Reranker feature importance" subtitle="Gain share of the trained XGBoost ranker (top 12)">
          {importance.length ? (
            <HBarChart ariaLabel="Reranker feature importance" data={importance} color="var(--c-rerank)" format={(v) => v.toFixed(3)} labelWidth={160} />
          ) : (
            <EmptyState title="Reranker not trained" description="The engine falls back to hybrid order." />
          )}
        </Card>
        <Card title="Entity labels in the corpus" subtitle="spaCy NER + gazetteer, all documents">
          {entityData.length ? <HBarChart ariaLabel="Entity label counts" data={entityData} color="var(--c-semantic)" labelWidth={120} /> : <EmptyState title="No entities" />}
        </Card>
      </div>

      <Card title="Live search metrics" subtitle="Since the API process started (in-memory, resets on restart; evaluation runs are not counted)">
        <div className="grid gap-6 md:grid-cols-[280px_1fr]">
          <dl className="grid grid-cols-2 gap-3 text-sm">
            {[
              ['Searches', fmtInt(s.total_searches)],
              ['Zero-result', fmtInt(s.zero_result_searches)],
              ['Avg latency', fmtMs(s.avg_latency_ms)],
              ['p95 latency', fmtMs(s.p95_latency_ms)],
              ['Avg candidates', s.avg_candidates === null ? 'N/A' : s.avg_candidates.toFixed(1)],
            ].map(([k, v]) => (
              <div key={k} className="rounded-md border border-line px-3 py-2">
                <dt className="text-[11px] text-faint">{k}</dt>
                <dd className="num font-semibold">{v}</dd>
              </div>
            ))}
          </dl>
          <div>
            <div className="mb-1 text-xs text-muted">Predicted intents of live searches</div>
            {intentData.length ? <HBarChart ariaLabel="Intent distribution" data={intentData} color="var(--accent)" /> : <EmptyState title="No searches yet" description="Run a few searches to populate live metrics." />}
          </div>
        </div>
      </Card>
    </div>
  )
}
