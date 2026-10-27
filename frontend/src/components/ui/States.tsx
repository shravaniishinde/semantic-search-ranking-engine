import { AlertTriangle, Inbox, Loader2, RefreshCw, WifiOff } from 'lucide-react'
import type { ReactNode } from 'react'
import { ApiError } from '../../services/api'

export function LoadingState({ label = 'Loading…', className = '' }: { label?: string; className?: string }) {
  return (
    <div role="status" className={`flex items-center justify-center gap-2 py-10 text-sm text-muted ${className}`}>
      <Loader2 size={16} className="animate-spin" aria-hidden />
      {label}
    </div>
  )
}

export function EmptyState({ title, description, action, icon }: { title: string; description?: ReactNode; action?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      <div className="mb-3 text-faint">{icon ?? <Inbox size={28} aria-hidden />}</div>
      <p className="text-sm font-medium text-fg">{title}</p>
      {description && <p className="mt-1 max-w-md text-sm text-muted">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function ErrorState({ error, onRetry, compact }: { error: unknown; onRetry?: () => void; compact?: boolean }) {
  const isNetwork = error instanceof ApiError && error.kind === 'network'
  const title = isNetwork ? 'Unable to connect to the search API' : 'The request failed'
  const message = error instanceof Error ? error.message : String(error)
  return (
    <div
      role="alert"
      className={`flex items-start gap-3 rounded-lg border border-bad/30 bg-bad-soft text-sm ${compact ? 'px-3 py-2' : 'px-4 py-3'}`}
    >
      {isNetwork ? <WifiOff size={16} className="mt-0.5 shrink-0 text-bad" /> : <AlertTriangle size={16} className="mt-0.5 shrink-0 text-bad" />}
      <div className="min-w-0 flex-1">
        <p className="font-medium text-bad">{title}</p>
        <p className="mt-0.5 break-words text-muted">
          {message}
          {isNetwork && ' Make sure the backend is running (uvicorn app.api.main:app --port 8000).'}
        </p>
      </div>
      {onRetry && (
        <button onClick={onRetry} className="inline-flex shrink-0 items-center gap-1 rounded border border-line bg-surface px-2 py-1 text-xs text-fg hover:bg-surface-2">
          <RefreshCw size={12} /> Retry
        </button>
      )}
    </div>
  )
}

/** Renders loading / error / content for a query-like object. */
export function QueryBoundary<T>({
  query,
  children,
  loadingLabel,
}: {
  query: { data: T | undefined; isLoading: boolean; error: unknown; refetch: () => unknown }
  children: (data: T) => ReactNode
  loadingLabel?: string
}) {
  if (query.isLoading) return <LoadingState label={loadingLabel} />
  if (query.error) return <ErrorState error={query.error} onRetry={() => query.refetch()} />
  if (query.data === undefined) return null
  return <>{children(query.data)}</>
}
