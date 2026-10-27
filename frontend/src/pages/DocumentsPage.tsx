import { ChevronLeft, ChevronRight, Copy, Database, FileText, Search, ScanSearch } from 'lucide-react'
import { useMemo, useState } from 'react'
import { ClusterBadge, DocumentStatusBadges } from '../components/documents/DocumentStatus'
import { Button, inputClass } from '../components/ui/Button'
import { StatCard, StatGrid } from '../components/ui/StatCard'
import { EmptyState, ErrorState, LoadingState } from '../components/ui/States'
import { useApp } from '../context/appState'
import { useAnalytics, useDocuments } from '../hooks/queries'
import { useDebounced } from '../hooks/useDebounced'
import type { DocumentQuery } from '../types/api'
import { fmtInt, humanize } from '../utils/format'

const PAGE_SIZE = 20

type StatusFilter = 'all' | 'indexed' | 'confirmed' | 'candidate' | 'outlier'

const STATUS_FILTERS: Record<StatusFilter, { label: string; query: DocumentQuery }> = {
  all: { label: 'All statuses', query: {} },
  indexed: { label: 'Indexed only', query: { include_excluded: false } },
  confirmed: { label: 'Confirmed duplicates', query: { is_duplicate: true } },
  candidate: { label: 'Duplicate candidates', query: { duplicate_candidate: true } },
  outlier: { label: 'Outliers', query: { is_outlier: true } },
}

export function DocumentsPage() {
  const { openDocument } = useApp()
  const analytics = useAnalytics()
  const [text, setText] = useState('')
  const [category, setCategory] = useState('')
  const [status, setStatus] = useState<StatusFilter>('all')
  const [page, setPage] = useState(0)
  const q = useDebounced(text.trim(), 250)

  const query = useMemo<DocumentQuery>(
    () => ({ offset: page * PAGE_SIZE, limit: PAGE_SIZE, q: q || undefined, category: category || undefined, ...STATUS_FILTERS[status].query }),
    [page, q, category, status],
  )
  const docs = useDocuments(query)
  const a = analytics.data
  const total = docs.data?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  const resetPage = <T,>(setter: (v: T) => void) => (v: T) => {
    setter(v)
    setPage(0)
  }

  return (
    <div className="space-y-4">
      <StatGrid>
        <StatCard label="Total documents" value={fmtInt(a?.document_count)} icon={FileText} loading={analytics.isLoading} />
        <StatCard label="Indexed" value={fmtInt(a?.indexed_count)} hint={a ? `${a.excluded_count} excluded` : undefined} icon={Database} loading={analytics.isLoading} />
        <StatCard label="Duplicate candidates" value={fmtInt(a?.duplicate_candidate_count)} icon={Copy} loading={analytics.isLoading} />
        <StatCard label="Confirmed duplicates" value={fmtInt(a?.duplicate_count)} icon={Copy} loading={analytics.isLoading} />
        <StatCard label="Outliers" value={fmtInt(a?.outlier_count)} icon={ScanSearch} loading={analytics.isLoading} />
      </StatGrid>

      <div className="rounded-lg border border-line bg-surface">
        <div className="flex flex-wrap items-center gap-2 border-b border-line p-3">
          <div className="relative min-w-56 flex-1">
            <Search size={14} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-faint" />
            <input
              aria-label="Filter by title or ID"
              value={text}
              onChange={(e) => resetPage(setText)(e.target.value)}
              placeholder="Filter by title or document ID…"
              className={`${inputClass} w-full pl-8`}
            />
          </div>
          <select aria-label="Category" value={category} onChange={(e) => resetPage(setCategory)(e.target.value)} className={inputClass}>
            <option value="">All categories</option>
            {Object.keys(a?.categories ?? {})
              .sort()
              .map((c) => (
                <option key={c} value={c}>
                  {humanize(c)} ({a?.categories[c]})
                </option>
              ))}
          </select>
          <select aria-label="Status" value={status} onChange={(e) => resetPage(setStatus)(e.target.value as StatusFilter)} className={inputClass}>
            {(Object.keys(STATUS_FILTERS) as StatusFilter[]).map((k) => (
              <option key={k} value={k}>
                {STATUS_FILTERS[k].label}
              </option>
            ))}
          </select>
        </div>

        {docs.isLoading && <LoadingState label="Loading documents…" />}
        {docs.error && (
          <div className="p-3">
            <ErrorState error={docs.error} onRetry={() => docs.refetch()} />
          </div>
        )}
        {docs.data && docs.data.documents.length === 0 && (
          <EmptyState
            title={a?.document_count === 0 ? 'No documents have been indexed yet.' : 'No documents match these filters.'}
            description={a?.document_count === 0 ? 'Use the Ingestion page to upload JSON or JSONL documents.' : undefined}
          />
        )}
        {docs.data && docs.data.documents.length > 0 && (
          <div className={`overflow-x-auto ${docs.isFetching ? 'opacity-70' : ''}`}>
            <table className="w-full text-[13px]">
              <thead className="border-b border-line text-left text-[11px] uppercase tracking-wide text-faint">
                <tr>
                  <th className="px-3 py-2 font-semibold">Document</th>
                  <th className="px-3 py-2 font-semibold">Category</th>
                  <th className="px-3 py-2 font-semibold">Source</th>
                  <th className="px-3 py-2 font-semibold">Cluster</th>
                  <th className="px-3 py-2 font-semibold">Entities</th>
                  <th className="px-3 py-2 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody>
                {docs.data.documents.map((d) => (
                  <tr key={d.document_id} onClick={() => openDocument(d.document_id)} className="cursor-pointer border-b border-line/60 last:border-0 hover:bg-surface-2/60">
                    <td className="max-w-md px-3 py-2">
                      <div className="truncate font-medium text-fg">{d.title}</div>
                      <div className="font-mono text-[11px] text-faint">{d.document_id}</div>
                    </td>
                    <td className="whitespace-nowrap px-3 py-2 text-muted">{humanize(d.category)}</td>
                    <td className="whitespace-nowrap px-3 py-2 text-muted">{d.source}</td>
                    <td className="px-3 py-2">
                      <ClusterBadge clusterId={d.cluster_id} />
                    </td>
                    <td className="num px-3 py-2 text-muted">{d.entities.length}</td>
                    <td className="px-3 py-2">
                      <DocumentStatusBadges doc={d} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="flex items-center justify-between border-t border-line px-3 py-2 text-xs text-muted">
          <span className="num">
            {total === 0 ? '0 documents' : `${page * PAGE_SIZE + 1}–${Math.min(total, (page + 1) * PAGE_SIZE)} of ${total}`}
          </span>
          <div className="flex items-center gap-1">
            <Button variant="ghost" aria-label="Previous page" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
              <ChevronLeft size={14} />
            </Button>
            <span className="num">
              Page {page + 1} / {pages}
            </span>
            <Button variant="ghost" aria-label="Next page" disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)}>
              <ChevronRight size={14} />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
