import { Boxes, Copy, Info, Layers, ScanSearch } from 'lucide-react'
import { useMemo, useState } from 'react'
import { HBarChart } from '../components/charts/Charts'
import { categoryColor } from '../utils/chartColors'
import { Badge } from '../components/ui/Badge'
import { Card } from '../components/ui/Card'
import { StatCard, StatGrid } from '../components/ui/StatCard'
import { EmptyState, ErrorState, LoadingState } from '../components/ui/States'
import { Tabs } from '../components/ui/Tabs'
import { useApp } from '../context/appState'
import { useClusters, useDocuments } from '../hooks/queries'
import type { ClusterAnalysis, ClusterInfo, Document } from '../types/api'
import { fmtNumber, humanize } from '../utils/format'

type TabKey = 'clusters' | 'duplicates' | 'outliers'

function CategoryBar({ cluster }: { cluster: ClusterInfo }) {
  const entries = Object.entries(cluster.categories).sort((a, b) => b[1] - a[1])
  return (
    <div>
      <div className="flex h-1.5 overflow-hidden rounded-full bg-surface-2">
        {entries.map(([cat, n]) => (
          <div key={cat} style={{ width: `${(n / cluster.size) * 100}%`, background: categoryColor(cat) }} />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
        {entries.map(([cat, n]) => (
          <span key={cat} className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-sm" style={{ background: categoryColor(cat) }} />
            {humanize(cat)} <span className="num text-faint">{n}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

function ClustersTab({ data }: { data: ClusterAnalysis }) {
  const { openDocument } = useApp()
  const [selected, setSelected] = useState<number | null>(data.clusters[0]?.cluster_id ?? null)
  const cluster = data.clusters.find((c) => c.cluster_id === selected)
  if (data.clusters.length === 0) return <EmptyState title="DBSCAN found no clusters" description="Every document is a noise point at the current eps / min_samples." />

  const chartData = data.clusters.map((c) => ({
    name: `#${c.cluster_id} ${humanize(c.dominant_category ?? '')}`,
    value: c.size,
    color: categoryColor(c.dominant_category ?? ''),
  }))

  return (
    <div className="grid items-start gap-4 xl:grid-cols-[420px_1fr]">
      <Card title="Cluster sizes" subtitle="Bar colour = dominant category · click to inspect">
        <HBarChart
          ariaLabel="Documents per cluster"
          data={chartData}
          labelWidth={190}
          onSelect={(name) => setSelected(Number(name.split(' ')[0].slice(1)))}
        />
      </Card>
      <div className="space-y-4">
        <div className="grid gap-2 sm:grid-cols-2 2xl:grid-cols-3">
          {data.clusters.map((c) => (
            <button
              key={c.cluster_id}
              onClick={() => setSelected(c.cluster_id)}
              aria-pressed={selected === c.cluster_id}
              className={`rounded-lg border bg-surface p-3 text-left transition-colors ${selected === c.cluster_id ? 'border-accent ring-1 ring-accent' : 'border-line hover:border-line-strong'}`}
            >
              <div className="mb-2 flex items-baseline justify-between">
                <span className="text-[13px] font-semibold">Cluster #{c.cluster_id}</span>
                <span className="num text-xs text-muted">{c.size} docs</span>
              </div>
              <CategoryBar cluster={c} />
            </button>
          ))}
        </div>
        {cluster && (
          <Card title={`Cluster #${cluster.cluster_id} · ${cluster.size} documents`} subtitle={`Dominant category: ${humanize(cluster.dominant_category ?? 'n/a')}`} bodyClassName="p-0">
            <ul>
              {cluster.document_ids.map((id, i) => (
                <li key={id} className="border-b border-line/60 last:border-0">
                  <button onClick={() => openDocument(id)} className="flex w-full items-baseline justify-between gap-3 px-4 py-2 text-left hover:bg-surface-2/60">
                    <span className="text-[13px] text-fg">{cluster.titles[i]}</span>
                    <span className="shrink-0 font-mono text-[11px] text-faint">{id}</span>
                  </button>
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>
    </div>
  )
}

function useClusterLookup(data: ClusterAnalysis) {
  return useMemo(() => {
    const clusterOf = new Map<string, number>()
    const titleOf = new Map<string, string>()
    for (const c of data.clusters)
      c.document_ids.forEach((id, i) => {
        clusterOf.set(id, c.cluster_id)
        titleOf.set(id, c.titles[i])
      })
    return { clusterOf, titleOf }
  }, [data])
}

function DocLink({ id, title }: { id: string; title?: string }) {
  const { openDocument } = useApp()
  return (
    <button onClick={() => openDocument(id)} className="text-left hover:text-accent">
      <div className="font-medium text-fg">{title ?? id}</div>
      {title && <div className="font-mono text-[11px] text-faint">{id}</div>}
    </button>
  )
}

function DuplicatesTab({ data, outlierDocs }: { data: ClusterAnalysis; outlierDocs: Document[] }) {
  const { clusterOf, titleOf } = useClusterLookup(data)
  for (const d of outlierDocs) titleOf.set(d.document_id, d.title)
  if (data.duplicate_pairs.length === 0) return <EmptyState title="No near-duplicate pairs" description="No document pair exceeds the duplicate-candidate cosine threshold." />
  const cluster = (id: string) => {
    const c = clusterOf.get(id)
    return c === undefined ? <Badge>noise</Badge> : <Badge tone="accent">#{c}</Badge>
  }
  return (
    <div className="space-y-3">
      <div className="flex items-start gap-2 text-xs text-muted">
        <Info size={14} className="mt-px shrink-0" />
        <p>
          A pair is a <b className="font-medium text-fg">candidate</b> when embedding cosine ≥ the candidate threshold. It is{' '}
          <b className="font-medium text-fg">confirmed</b> only if cosine ≥ the confirm threshold and token-set Jaccard ≥ the lexical threshold;
          the later document is then excluded from the index (never deleted).
        </p>
      </div>
      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        <table className="w-full text-[13px]">
          <thead className="border-b border-line text-left text-[11px] uppercase tracking-wide text-faint">
            <tr>
              <th className="px-3 py-2 font-semibold">Canonical document</th>
              <th className="px-3 py-2 font-semibold">Similar document</th>
              <th className="px-3 py-2 text-right font-semibold">Cosine</th>
              <th className="px-3 py-2 text-right font-semibold">Token Jaccard</th>
              <th className="px-3 py-2 font-semibold">Clusters</th>
              <th className="px-3 py-2 font-semibold">Status</th>
            </tr>
          </thead>
          <tbody>
            {data.duplicate_pairs.map((p) => (
              <tr key={`${p.document_id}-${p.other_document_id}`} className="border-b border-line/60 last:border-0">
                <td className="px-3 py-2">
                  <DocLink id={p.document_id} title={titleOf.get(p.document_id)} />
                </td>
                <td className="px-3 py-2">
                  <DocLink id={p.other_document_id} title={titleOf.get(p.other_document_id)} />
                </td>
                <td className="num px-3 py-2 text-right">{fmtNumber(p.cosine_similarity, 4)}</td>
                <td className="num px-3 py-2 text-right">{fmtNumber(p.jaccard_similarity, 4)}</td>
                <td className="px-3 py-2">
                  <div className="flex gap-1">
                    {cluster(p.document_id)}
                    {cluster(p.other_document_id)}
                  </div>
                </td>
                <td className="px-3 py-2">{p.confirmed ? <Badge tone="bad">confirmed duplicate</Badge> : <Badge tone="warn">candidate only</Badge>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function OutliersTab({ outliers, loading, error }: { outliers: Document[]; loading: boolean; error: unknown }) {
  const { openDocument } = useApp()
  return (
    <div className="space-y-3">
      <div className="flex items-start gap-2 rounded-md border border-info/30 bg-info-soft px-3 py-2 text-[13px] text-fg">
        <Info size={15} className="mt-0.5 shrink-0 text-info" />
        <p>
          <b className="font-semibold">An outlier does not automatically mean a bad document.</b> DBSCAN labels a document as noise when it has no dense
          neighbourhood of similar documents. Off-topic or unusually specific documents land here. Outliers stay indexed and searchable.
        </p>
      </div>
      {loading && <LoadingState />}
      {!!error && <ErrorState error={error} />}
      {!loading && !error && outliers.length === 0 && <EmptyState title="No outliers" />}
      {outliers.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-line bg-surface">
          <table className="w-full text-[13px]">
            <thead className="border-b border-line text-left text-[11px] uppercase tracking-wide text-faint">
              <tr>
                <th className="px-3 py-2 font-semibold">Document</th>
                <th className="px-3 py-2 font-semibold">Category</th>
                <th className="px-3 py-2 font-semibold">Source</th>
                <th className="px-3 py-2 font-semibold">Cluster</th>
                <th className="px-3 py-2 font-semibold">Index status</th>
              </tr>
            </thead>
            <tbody>
              {outliers.map((d) => (
                <tr key={d.document_id} onClick={() => openDocument(d.document_id)} className="cursor-pointer border-b border-line/60 last:border-0 hover:bg-surface-2/60">
                  <td className="px-3 py-2">
                    <div className="font-medium text-fg">{d.title}</div>
                    <div className="font-mono text-[11px] text-faint">{d.document_id}</div>
                  </td>
                  <td className="px-3 py-2 text-muted">{humanize(d.category)}</td>
                  <td className="px-3 py-2 text-muted">{d.source}</td>
                  <td className="px-3 py-2">
                    <Badge>noise (-1)</Badge>
                  </td>
                  <td className="px-3 py-2">{d.indexed ? <Badge tone="ok">indexed</Badge> : <Badge tone="bad">{d.exclusion_reason ?? 'excluded'}</Badge>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export function ClustersPage() {
  const clusters = useClusters()
  const [tab, setTab] = useState<TabKey>('clusters')
  // Outlier titles/categories are not part of /clusters; fetch only those documents.
  const outliers = useDocuments({ is_outlier: true, limit: 200 }, tab !== 'clusters')

  if (clusters.isLoading) return <LoadingState label="Loading cluster analysis…" />
  if (clusters.error) return <ErrorState error={clusters.error} onRetry={() => clusters.refetch()} />
  const data = clusters.data
  if (!data) return null
  if (data.total_documents === 0) return <EmptyState title="No documents have been indexed yet." />

  return (
    <div className="space-y-4">
      <StatGrid>
        <StatCard label="Clusters" value={data.cluster_count} hint={`${data.clustered_documents} of ${data.total_documents} documents`} icon={Boxes} />
        <StatCard label="Duplicate candidates" value={data.duplicate_candidate_count} hint="documents in a high-similarity pair" icon={Copy} />
        <StatCard label="Confirmed duplicates" value={data.confirmed_duplicate_count} hint="excluded from index" icon={Copy} />
        <StatCard label="Outliers" value={data.outlier_count} hint="DBSCAN noise points" icon={ScanSearch} />
        <StatCard label="DBSCAN parameters" value={<span className="text-base">eps {data.eps} · min {data.min_samples}</span>} hint="cosine, mean-centred embeddings" icon={Layers} />
      </StatGrid>
      <Tabs
        ariaLabel="Cluster analysis views"
        value={tab}
        onChange={setTab}
        items={[
          { key: 'clusters', label: 'Clusters', count: data.cluster_count },
          { key: 'duplicates', label: 'Duplicates', count: data.duplicate_pairs.length },
          { key: 'outliers', label: 'Outliers', count: data.outlier_count },
        ]}
      />
      {tab === 'clusters' && <ClustersTab data={data} />}
      {tab === 'duplicates' && <DuplicatesTab data={data} outlierDocs={outliers.data?.documents ?? []} />}
      {tab === 'outliers' && <OutliersTab outliers={outliers.data?.documents ?? []} loading={outliers.isLoading} error={outliers.error} />}
    </div>
  )
}
