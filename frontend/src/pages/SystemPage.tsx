import { Info } from 'lucide-react'
import type { ReactNode } from 'react'
import { StatusDot } from '../components/ui/Badge'
import { Card } from '../components/ui/Card'
import { ErrorState, LoadingState } from '../components/ui/States'
import { useHealth } from '../hooks/queries'
import { API_BASE_URL } from '../services/api'
import { describeBm25, type Tone } from '../utils/health'
import { fmtInt, fmtMs } from '../utils/format'

function StatusRow({ name, tone, status, detail }: { name: string; tone: Tone; status: string; detail?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-line/60 px-4 py-3 last:border-0">
      <div>
        <div className="text-[13px] font-medium">{name}</div>
        {detail && <div className="mt-0.5 text-xs text-muted">{detail}</div>}
      </div>
      <div className="flex shrink-0 items-center gap-2 text-[13px]">
        <StatusDot tone={tone} />
        <span>{status}</span>
      </div>
    </div>
  )
}

export function SystemPage() {
  const health = useHealth()
  if (health.isLoading) return <LoadingState label="Checking backend health…" />
  if (health.error)
    return (
      <div className="space-y-4">
        <ErrorState error={health.error} onRetry={() => health.refetch()} />
        <Card title="API">
          <StatusRow name="API endpoint" tone="bad" status="Unreachable" detail={API_BASE_URL} />
        </Card>
      </div>
    )
  if (!health.data) return null
  const { data: h, roundTripMs } = health.data
  const bm25 = describeBm25(h)
  const ready = h.status === 'ok' && h.indexed > 0
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-4">
        {bm25.banner && (
          <div
            className={`flex items-start gap-2 rounded-lg border px-4 py-3 text-[13px] ${
              bm25.banner.tone === 'warn' ? 'border-warn/30 bg-warn-soft' : 'border-info/30 bg-info-soft'
            }`}
          >
            <Info size={16} className={`mt-0.5 shrink-0 ${bm25.banner.tone === 'warn' ? 'text-warn' : 'text-info'}`} />
            <div>
              <p className="font-medium">{bm25.banner.title}</p>
              <p className="mt-0.5 text-muted">{bm25.banner.message}</p>
            </div>
          </div>
        )}
        <Card title="Services" bodyClassName="p-0">
          <StatusRow name="API" tone={h.status === 'ok' ? 'ok' : 'bad'} status={h.status === 'ok' ? 'Healthy' : h.status} detail={`${API_BASE_URL} · round trip ${fmtMs(roundTripMs)}`} />
          <StatusRow name="Search engine" tone={ready ? 'ok' : 'warn'} status={ready ? 'Ready' : 'No indexed documents'} detail={`${fmtInt(h.documents)} documents, ${fmtInt(h.indexed)} indexed`} />
          <StatusRow name="FAISS dense index" tone={h.faiss_vectors > 0 ? 'ok' : 'warn'} status={h.faiss_vectors > 0 ? 'Loaded' : 'Empty'} detail={`${fmtInt(h.faiss_vectors)} vectors · IndexFlatIP (exact cosine)`} />
          <StatusRow name="BM25 lexical index" tone={bm25.tone} status={bm25.status} detail={`${fmtInt(h.bm25_documents)} documents · requested backend: ${h.bm25_backend_requested}`} />
        </Card>
      </div>
      <div className="space-y-4">
        <Card title="Models" bodyClassName="p-0">
          <StatusRow name="Embedding model" tone="ok" status="Loaded" detail={h.embedding_model} />
          <StatusRow name="Named entity recognition" tone={h.ner_backend === 'spacy' ? 'ok' : 'warn'} status={h.ner_backend === 'spacy' ? 'spaCy + gazetteer' : 'Gazetteer only'} />
          <StatusRow name="Intent classifier" tone={h.intent_model === 'xgboost' ? 'ok' : 'warn'} status={h.intent_model === 'xgboost' ? 'XGBoost trained' : 'Rule fallback'} />
          <StatusRow name="Reranker" tone={h.reranker === 'xgboost_ltr' ? 'ok' : 'warn'} status={h.reranker === 'xgboost_ltr' ? 'XGBoost LTR trained' : 'Hybrid-order fallback'} />
        </Card>
        <Card title="Raw /health response">
          <pre className="overflow-x-auto text-xs text-muted">{JSON.stringify(h, null, 2)}</pre>
        </Card>
      </div>
    </div>
  )
}
