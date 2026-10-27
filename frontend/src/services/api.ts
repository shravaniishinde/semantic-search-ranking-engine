// Single place where the dashboard talks to the FastAPI backend.
import type {
  Analytics,
  ClusterAnalysis,
  Document,
  DocumentIn,
  DocumentListResponse,
  DocumentQuery,
  EvaluationResponse,
  Health,
  IngestionReport,
  SearchRequest,
  SearchResponse,
} from '../types/api'

export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/+$/, '')

/** `kind` lets the UI distinguish "backend is down" from "backend rejected the request". */
export class ApiError extends Error {
  readonly status: number | null
  readonly kind: 'network' | 'http'

  constructor(message: string, status: number | null, kind: 'network' | 'http') {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.kind = kind
  }
}

type QueryValue = string | number | boolean | null | undefined

export function buildUrl(path: string, params?: Record<string, QueryValue>): string {
  const url = new URL(path.replace(/^\//, ''), `${API_BASE_URL}/`)
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== '') url.searchParams.set(key, String(value))
  }
  return url.toString()
}

function describeHttpError(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  // FastAPI 422 validation errors: [{loc, msg, ...}]
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: string; loc?: unknown[] }
    const field = Array.isArray(first.loc) ? first.loc.filter((p) => p !== 'body').join('.') : ''
    return field ? `${field}: ${first.msg ?? 'invalid value'}` : (first.msg ?? 'Invalid request')
  }
  return `Request failed with status ${status}`
}

async function request<T>(path: string, init?: RequestInit & { params?: Record<string, QueryValue> }): Promise<T> {
  const { params, ...rest } = init ?? {}
  let response: Response
  try {
    response = await fetch(buildUrl(path, params), {
      ...rest,
      headers: rest.body ? { 'Content-Type': 'application/json', ...rest.headers } : rest.headers,
    })
  } catch {
    throw new ApiError(`Unable to connect to the search API at ${API_BASE_URL}.`, null, 'network')
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw new ApiError(describeHttpError(response.status, body), response.status, 'http')
  return body as T
}

export interface Timed<T> {
  data: T
  roundTripMs: number
}

export async function getHealth(): Promise<Timed<Health>> {
  const started = performance.now()
  const data = await request<Health>('/health')
  return { data, roundTripMs: performance.now() - started }
}

export function searchDocuments(req: SearchRequest): Promise<SearchResponse> {
  return request<SearchResponse>('/search', { method: 'POST', body: JSON.stringify(req) })
}

export function getDocuments(query: DocumentQuery = {}): Promise<DocumentListResponse> {
  return request<DocumentListResponse>('/documents', { params: { ...query } })
}

export function getDocument(documentId: string): Promise<Document> {
  return request<Document>(`/documents/${encodeURIComponent(documentId)}`)
}

export function getAnalytics(): Promise<Analytics> {
  return request<Analytics>('/analytics')
}

export function getClusters(): Promise<ClusterAnalysis> {
  return request<ClusterAnalysis>('/clusters')
}

export function ingestDocuments(documents: DocumentIn[]): Promise<IngestionReport> {
  return request<IngestionReport>('/documents/ingest', { method: 'POST', body: JSON.stringify({ documents }) })
}

export function getEvaluation(): Promise<EvaluationResponse> {
  return request<EvaluationResponse>('/evaluation')
}

export function runEvaluation(): Promise<EvaluationResponse> {
  return request<EvaluationResponse>('/evaluation/run', { method: 'POST' })
}
