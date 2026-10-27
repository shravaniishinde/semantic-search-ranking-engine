import type { ReactNode } from 'react'
import { useApp } from '../../context/appState'
import { useDocument } from '../../hooks/queries'
import type { Document, SearchResultItem } from '../../types/api'
import { NA, fmtNumber, humanize } from '../../utils/format'
import { RankingExplanation } from '../search/RankingExplanation'
import { SectionLabel } from '../ui/Card'
import { Drawer } from '../ui/Drawer'
import { ErrorState, LoadingState } from '../ui/States'
import { DocumentStatusBadges, ClusterBadge } from './DocumentStatus'
import { EntityChips } from './EntityChips'

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wide text-faint">{label}</dt>
      <dd className="mt-0.5 text-[13px] text-fg">{children}</dd>
    </div>
  )
}

export function DocumentDetail({ doc, result }: { doc: Document; result?: SearchResultItem }) {
  const { openDocument } = useApp()
  return (
    <div className="space-y-6">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3">
        <Field label="Category">{humanize(doc.category)}</Field>
        <Field label="Source">{doc.source}</Field>
        <Field label="Date">{doc.created_at ?? NA}</Field>
        <Field label="Cluster">
          <ClusterBadge clusterId={doc.cluster_id} />
        </Field>
        <Field label="Status">
          <DocumentStatusBadges doc={doc} />
        </Field>
        <Field label="Embedding">{doc.embedding ? `${doc.embedding.dimension}-d · normalised` : NA}</Field>
      </dl>

      {(doc.exclusion_reason || doc.duplicate_of) && (
        <div className="rounded-md border border-warn/30 bg-warn-soft px-3 py-2 text-[13px]">
          {doc.exclusion_reason && <p className="text-fg">Excluded from the index: {doc.exclusion_reason}.</p>}
          {doc.duplicate_of && (
            <p className="mt-1 text-muted">
              Canonical document:{' '}
              <button className="font-mono text-accent hover:underline" onClick={() => openDocument(doc.duplicate_of as string)}>
                {doc.duplicate_of}
              </button>
            </p>
          )}
        </div>
      )}
      {doc.is_outlier && (
        <p className="rounded-md border border-info/30 bg-info-soft px-3 py-2 text-[13px] text-fg">
          DBSCAN marked this document as an outlier (no dense neighbourhood of similar documents). That does not mean it is a bad
          document; it stays indexed.
        </p>
      )}

      {result && (
        <div>
          <SectionLabel>Retrieval scores for the current query</SectionLabel>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {[
              ['Final', fmtNumber(result.final_score), 'text-rerank'],
              ['BM25', result.bm25_score === null ? 'not retrieved' : fmtNumber(result.bm25_score), 'text-bm25'],
              ['Semantic', result.semantic_score === null ? 'not retrieved' : fmtNumber(result.semantic_score), 'text-semantic'],
              ['Hybrid (RRF)', fmtNumber(result.hybrid_score, 4), 'text-hybrid'],
            ].map(([label, value, color]) => (
              <div key={label} className="rounded-md border border-line px-3 py-2">
                <div className="text-[11px] text-faint">{label}</div>
                <div className={`num text-sm font-semibold ${color}`}>{value}</div>
              </div>
            ))}
          </div>
          <div className="mt-4">
            <RankingExplanation result={result} />
          </div>
        </div>
      )}

      <div>
        <SectionLabel>Entities ({doc.entities.length})</SectionLabel>
        <EntityChips entities={doc.entities} />
      </div>

      <div>
        <SectionLabel>Content</SectionLabel>
        <p className="whitespace-pre-line text-[13.5px] leading-relaxed text-fg">{doc.content}</p>
      </div>

      {Object.keys(doc.metadata).length > 0 && (
        <div>
          <SectionLabel>Metadata</SectionLabel>
          <pre className="overflow-x-auto rounded-md bg-surface-2 p-3 text-xs">{JSON.stringify(doc.metadata, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

export function DocumentDrawer() {
  const { drawer, closeDocument } = useApp()
  const query = useDocument(drawer?.documentId ?? null)
  const doc = query.data
  return (
    <Drawer
      open={!!drawer}
      onClose={closeDocument}
      title={
        <div>
          <div className="font-mono text-xs text-faint">{drawer?.documentId}</div>
          <h2 className="mt-0.5 text-base font-semibold leading-snug">{doc?.title ?? drawer?.result?.title ?? 'Document'}</h2>
        </div>
      }
    >
      {query.isLoading && <LoadingState label="Loading document…" />}
      {query.error && <ErrorState error={query.error} onRetry={() => query.refetch()} />}
      {doc && <DocumentDetail doc={doc} result={drawer?.result} />}
    </Drawer>
  )
}
