import { useMemo, useState } from 'react'
import { useApp } from '../../context/appState'
import type { HybridHit, RetrievalHit, SearchResponse } from '../../types/api'
import { fmtNumber } from '../../utils/format'
import { EmptyState } from '../ui/States'
import { Tabs } from '../ui/Tabs'
import { ResultCard } from './ResultCard'

type TabKey = 'final' | 'bm25' | 'semantic' | 'hybrid'

function FinalRankCell({ rank }: { rank: number | undefined }) {
  return rank ? <span className="num font-medium text-rerank">#{rank}</span> : <span className="text-faint">—</span>
}

function useFinalRanks(response: SearchResponse) {
  return useMemo(() => new Map(response.results.map((r) => [r.document_id, r.rank])), [response])
}

const th = 'px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-faint'
const td = 'px-3 py-2'

function TitleCell({ id, title }: { id: string; title: string }) {
  const { openDocument } = useApp()
  return (
    <td className={td}>
      <button onClick={() => openDocument(id)} className="text-left font-medium text-fg hover:text-accent">
        {title}
      </button>
      <div className="font-mono text-[11px] text-faint">{id}</div>
    </td>
  )
}

function RetrievalTable({ hits, scoreLabel, scoreClass, finalRanks }: { hits: RetrievalHit[]; scoreLabel: string; scoreClass: string; finalRanks: Map<string, number> }) {
  if (hits.length === 0) return <EmptyState title="No documents returned" description="This retriever found no matching documents for the query." />
  return (
    <table className="w-full text-[13px]">
      <thead className="border-b border-line">
        <tr>
          <th className={`${th} w-14`}>Rank</th>
          <th className={th}>Document</th>
          <th className={`${th} text-right`}>{scoreLabel}</th>
          <th className={`${th} text-right`}>Final rank</th>
        </tr>
      </thead>
      <tbody>
        {hits.map((h) => (
          <tr key={h.document_id} className="border-b border-line/60 last:border-0 hover:bg-surface-2/60">
            <td className={`${td} num text-muted`}>#{h.rank}</td>
            <TitleCell id={h.document_id} title={h.title} />
            <td className={`${td} num text-right font-semibold ${scoreClass}`}>{fmtNumber(h.score)}</td>
            <td className={`${td} text-right`}>
              <FinalRankCell rank={finalRanks.get(h.document_id)} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function HybridTable({ hits, finalRanks }: { hits: HybridHit[]; finalRanks: Map<string, number> }) {
  if (hits.length === 0) return <EmptyState title="No fused candidates" />
  return (
    <table className="w-full text-[13px]">
      <thead className="border-b border-line">
        <tr>
          <th className={`${th} w-14`}>Rank</th>
          <th className={th}>Document</th>
          <th className={`${th} text-right`}>BM25</th>
          <th className={`${th} text-right`}>Semantic</th>
          <th className={`${th} text-right`}>RRF score</th>
          <th className={`${th} text-right`}>Final rank</th>
        </tr>
      </thead>
      <tbody>
        {hits.map((h, i) => (
          <tr key={h.document_id} className="border-b border-line/60 last:border-0 hover:bg-surface-2/60">
            <td className={`${td} num text-muted`}>#{i + 1}</td>
            <TitleCell id={h.document_id} title={h.title} />
            <td className={`${td} num text-right`}>
              {h.bm25_rank === null ? <span className="text-faint">—</span> : <span className="text-bm25">{fmtNumber(h.bm25_score)} <span className="text-faint">#{h.bm25_rank}</span></span>}
            </td>
            <td className={`${td} num text-right`}>
              {h.semantic_rank === null ? <span className="text-faint">—</span> : <span className="text-semantic">{fmtNumber(h.semantic_score)} <span className="text-faint">#{h.semantic_rank}</span></span>}
            </td>
            <td className={`${td} num text-right font-semibold text-hybrid`}>{fmtNumber(h.hybrid_score, 4)}</td>
            <td className={`${td} text-right`}>
              <FinalRankCell rank={finalRanks.get(h.document_id)} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function ResultTabs({ response }: { response: SearchResponse }) {
  const [tab, setTab] = useState<TabKey>('final')
  const finalRanks = useFinalRanks(response)
  const r = response.retrieval
  const items = [
    { key: 'final' as const, label: 'Final ranking', count: response.results.length, dotClass: 'bg-rerank' },
    { key: 'bm25' as const, label: 'BM25', count: r?.bm25.length, dotClass: 'bg-bm25' },
    { key: 'semantic' as const, label: 'Semantic', count: r?.semantic.length, dotClass: 'bg-semantic' },
    { key: 'hybrid' as const, label: 'Hybrid (RRF)', count: r?.hybrid.length, dotClass: 'bg-hybrid' },
  ]
  return (
    <section aria-label="Results">
      <Tabs items={items} value={tab} onChange={setTab} ariaLabel="Result lists" />
      <div className="mt-3">
        {tab === 'final' &&
          (response.results.length === 0 ? (
            <EmptyState title="No results" description="Neither retriever returned a candidate. Try a broader query or remove filters." />
          ) : (
            <div className="space-y-3">
              {response.results.map((res) => (
                <ResultCard key={res.document_id} result={res} />
              ))}
            </div>
          ))}
        {tab !== 'final' && !r && <EmptyState title="Retrieval details were not requested" />}
        {tab !== 'final' && r && (
          <div className="overflow-x-auto rounded-lg border border-line bg-surface">
            {tab === 'bm25' && <RetrievalTable hits={r.bm25} scoreLabel={`BM25 score (${r.bm25_backend})`} scoreClass="text-bm25" finalRanks={finalRanks} />}
            {tab === 'semantic' && <RetrievalTable hits={r.semantic} scoreLabel="Cosine similarity" scoreClass="text-semantic" finalRanks={finalRanks} />}
            {tab === 'hybrid' && <HybridTable hits={r.hybrid} finalRanks={finalRanks} />}
          </div>
        )}
        {tab !== 'final' && r && (
          <p className="mt-2 text-xs text-faint">
            Showing the top {response.results.length || 'K'} of each list. “Final rank” shows where the document ended up after XGBoost reranking (— = not in the
            final top-K).
          </p>
        )}
      </div>
    </section>
  )
}
