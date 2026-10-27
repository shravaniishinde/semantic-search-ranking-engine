import { Search } from 'lucide-react'
import { useApp } from '../context/appState'
import { useAnalytics, useSearch } from '../hooks/queries'
import type { SearchRequest } from '../types/api'
import { LatencyBreakdown } from '../components/search/LatencyBreakdown'
import { PipelineView } from '../components/search/PipelineView'
import { QueryInsights } from '../components/search/QueryInsights'
import { ResultTabs } from '../components/search/ResultTabs'
import { SearchBar, type SearchControls } from '../components/search/SearchBar'
import { EmptyState, ErrorState, LoadingState } from '../components/ui/States'

// Suggestions only pre-fill the search box; results always come from the backend.
const EXAMPLES = [
  'How does Kubernetes handle container orchestration?',
  'How can cloud infrastructure be secured?',
  'How do databases improve query performance?',
  'stop attackers from guessing user passwords',
  'CVE-2021-44228',
]

function toRequest(c: SearchControls): SearchRequest {
  const filters = c.category || c.source ? { category: c.category || null, source: c.source || null } : null
  return { query: c.query.trim(), top_k: c.topK, filters, include_retrieval: true }
}

export function SearchPage() {
  const { session, setSession } = useApp()
  const analytics = useAnalytics()
  const search = useSearch()

  const initial: SearchControls = {
    query: session?.request.query ?? '',
    topK: session?.request.top_k ?? 10,
    category: session?.request.filters?.category ?? '',
    source: session?.request.filters?.source ?? '',
  }

  const run = (controls: SearchControls) => {
    const request = toRequest(controls)
    search.mutate(request, { onSuccess: (response) => setSession({ request, response }) })
  }

  const categories = Object.keys(analytics.data?.categories ?? {}).sort()
  const sources = Object.keys(analytics.data?.sources ?? {}).sort()
  const response = session?.response

  return (
    <div className="space-y-4">
      <SearchBar
        // Remount when an example is picked so the input shows it.
        key={session?.request.query ?? 'empty'}
        initial={initial}
        categories={categories}
        sources={sources}
        loading={search.isPending}
        onSearch={run}
        onControlsChange={(c) => session && c.query.trim() && run(c)}
        onClear={() => {
          search.reset()
          setSession(null)
        }}
      />

      {search.isError && (
        <ErrorState
          error={search.error}
          onRetry={() => {
            const last = search.variables
            if (last) search.mutate(last, { onSuccess: (r) => setSession({ request: last, response: r }) })
          }}
        />
      )}

      {search.isPending && !response && <LoadingState label="Running NER, intent, BM25, FAISS, fusion and reranking…" />}

      {!response && !search.isPending && !search.isError && (
        <div className="rounded-lg border border-dashed border-line-strong bg-surface">
          <EmptyState
            icon={<Search size={28} />}
            title="Search the document collection to see ranked results."
            description="Each query runs through spaCy NER, XGBoost intent classification, BM25 and FAISS retrieval, Reciprocal Rank Fusion and an XGBoost learning-to-rank reranker. Every stage is shown below."
            action={
              <div className="flex max-w-2xl flex-wrap justify-center gap-2">
                {EXAMPLES.map((q) => (
                  <button
                    key={q}
                    onClick={() => run({ query: q, topK: 10, category: '', source: '' })}
                    className="rounded-full border border-line bg-surface-2 px-3 py-1 text-xs text-muted hover:border-accent hover:text-fg"
                  >
                    {q}
                  </button>
                ))}
              </div>
            }
          />
        </div>
      )}

      {response && (
        <div className={`space-y-4 transition-opacity ${search.isPending ? 'opacity-60' : ''}`} aria-busy={search.isPending}>
          <QueryInsights response={response} />
          <PipelineView response={response} />
          <div className="grid gap-4 xl:grid-cols-[1fr_320px]">
            <div className="min-w-0">
              <ResultTabs key={response.query + response.latency_ms} response={response} />
            </div>
            <div className="space-y-4">
              <LatencyBreakdown response={response} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
