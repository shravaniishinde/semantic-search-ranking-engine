import { ChevronRight } from 'lucide-react'
import type { ReactNode } from 'react'
import type { SearchResponse } from '../../types/api'
import { NA, fmtMs, humanize } from '../../utils/format'

interface StageProps {
  name: string
  metric: ReactNode
  detail?: ReactNode
  ms?: number
  accent?: string
}

function Stage({ name, metric, detail, ms, accent = 'border-t-line-strong' }: StageProps) {
  return (
    <div className={`min-w-0 flex-1 rounded-md border border-line border-t-2 bg-surface px-3 py-2 ${accent}`}>
      <div className="truncate text-[11px] font-semibold uppercase tracking-wide text-faint">{name}</div>
      <div className="num mt-0.5 truncate text-sm font-semibold text-fg">{metric}</div>
      {detail && <div className="truncate text-[11px] text-muted">{detail}</div>}
      <div className="num mt-1 text-[11px] text-faint">{ms === undefined ? NA : fmtMs(ms)}</div>
    </div>
  )
}

function Arrow() {
  return <ChevronRight size={16} className="hidden shrink-0 self-center text-faint xl:block" aria-hidden />
}

/** Visualises the request path through app/services/search_service.py with real counts and timings. */
export function PipelineView({ response }: { response: SearchResponse }) {
  const s = response.stage_latency_ms
  const counts = response.retrieval_counts
  const backend = response.retrieval?.bm25_backend ?? NA
  const featureCount = response.results[0] ? Object.keys(response.results[0].features).length : null
  return (
    <section aria-label="Retrieval pipeline" className="rounded-lg border border-line bg-surface-2/50 p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-[13px] font-semibold">Retrieval pipeline</h2>
        <span className="text-[11px] text-faint">Counts and timings from this request</span>
      </div>
      <div className="grid grid-cols-2 items-start gap-2 md:grid-cols-4 xl:flex xl:items-center xl:gap-1.5">
        <Stage name="Query" metric={`${response.query.split(/\s+/).length} words`} detail="cleaned" ms={s.preprocessing} />
        <Arrow />
        <Stage name="NER" metric={`${response.entities.length} ${response.entities.length === 1 ? 'entity' : 'entities'}`} detail="spaCy" ms={s.ner} />
        <Arrow />
        <Stage name="Intent" metric={humanize(response.intent.label)} detail={response.intent.source} ms={s.intent} />
        <Arrow />
        <Stage name="Embedding" metric="query vector" detail="MiniLM" ms={s.query_embedding} />
        <Arrow />
        <div className="col-span-2 flex min-w-0 flex-[2] flex-col gap-1.5 md:row-span-2 xl:col-span-1">
          <Stage name="BM25" metric={`${counts.bm25 ?? NA} candidates`} detail={backend === 'memory' ? 'in-memory' : backend} ms={s.bm25} accent="border-t-bm25" />
          <Stage name="FAISS" metric={`${counts.semantic ?? NA} candidates`} detail="cosine, exact" ms={s.semantic} accent="border-t-semantic" />
        </div>
        <Arrow />
        <Stage name="RRF fusion" metric={`${counts.hybrid ?? NA} merged`} detail="rank-based" ms={s.fusion} accent="border-t-hybrid" />
        <Arrow />
        <Stage name="Features" metric={featureCount === null ? NA : `${featureCount} per doc`} detail="per candidate" ms={s.features} />
        <Arrow />
        <Stage name="Rerank" metric={`${response.results.length} results`} detail={response.ranker === 'xgboost_ltr' ? 'XGBoost LTR' : response.ranker} ms={s.rerank} accent="border-t-rerank" />
      </div>
    </section>
  )
}
