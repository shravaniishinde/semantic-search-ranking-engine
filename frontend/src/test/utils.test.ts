import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, buildUrl, getDocuments, searchDocuments } from '../services/api'
import { featureRows } from '../utils/features'
import { describeBm25 } from '../utils/health'
import { parseDocuments } from '../utils/ingestion'
import { health, searchResponse } from './fixtures'

describe('API client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('builds URLs from the configured base and drops empty params', () => {
    const url = new URL(buildUrl('/documents', { limit: 20, q: '', category: undefined, is_outlier: true }))
    expect(url.pathname).toBe('/documents')
    expect(url.searchParams.get('limit')).toBe('20')
    expect(url.searchParams.get('is_outlier')).toBe('true')
    expect(url.searchParams.has('q')).toBe(false)
    expect(url.searchParams.has('category')).toBe(false)
  })

  it('turns network failures into a network ApiError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(getDocuments()).rejects.toMatchObject({ kind: 'network', status: null })
  })

  it('surfaces FastAPI error details', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'query contains no searchable text' }), { status: 400 })))
    const err = await searchDocuments({ query: '<p></p>' }).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err).toMatchObject({ status: 400, kind: 'http', message: 'query contains no searchable text' })
  })

  it('summarises FastAPI 422 validation errors', async () => {
    const body = { detail: [{ loc: ['body', 'top_k'], msg: 'Input should be less than or equal to 50' }] }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 422 })))
    await expect(searchDocuments({ query: 'x', top_k: 99 })).rejects.toThrow('top_k: Input should be less than or equal to 50')
  })

  it('posts search requests as JSON', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(searchResponse), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await searchDocuments({ query: 'aws', top_k: 5 })
    const [, init] = fetchMock.mock.calls[0]
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ query: 'aws', top_k: 5 })
  })
})

describe('feature formatting', () => {
  it('shows missing ranks as "not retrieved" and folds intent one-hots', () => {
    const rows = featureRows(searchResponse.results[1].features)
    const byKey = Object.fromEntries(rows.map((r) => [r.key, r]))
    expect(byKey.bm25_rank.display).toBe('not retrieved')
    expect(byKey.semantic_rank.display).toBe('#1')
    expect(byKey.intent_onehot.display).toBe('Technical')
    expect(rows.some((r) => r.key === 'intent_technical')).toBe(false)
    expect(byKey.doc_length_log.display).toBe('69 words')
    expect(byKey.query_term_coverage.bar).toBeCloseTo(0.667, 2)
  })
})

describe('ingestion parsing', () => {
  it('parses JSONL and reports bad lines', () => {
    const r = parseDocuments('{"document_id":"a","content":"x"}\nnot json\n\n[1]\n', 'docs.jsonl')
    expect(r.format).toBe('jsonl')
    expect(r.documents.map((d) => d.document_id)).toEqual(['a'])
    expect(r.errors).toEqual(['Line 2: invalid JSON', 'Line 4: expected a JSON object'])
  })

  it('parses a JSON list and a {"documents": [...]} wrapper', () => {
    expect(parseDocuments('[{"title":"a"},{"title":"b"}]', 'x.json').documents).toHaveLength(2)
    expect(parseDocuments('{"documents":[{"title":"a"}]}', 'x.json').documents).toHaveLength(1)
  })

  it('reports invalid and empty files', () => {
    expect(parseDocuments('', 'x.json').errors).toEqual(['The file is empty.'])
    expect(parseDocuments('{oops', 'x.json').errors[0]).toMatch(/^Invalid JSON/)
  })
})

describe('BM25 backend status', () => {
  it('distinguishes configured, fallback, healthy ES and ES outage', () => {
    expect(describeBm25({ ...health, bm25_backend_requested: 'memory' }).banner).toBeNull()
    expect(describeBm25(health)).toMatchObject({ tone: 'info', status: 'In-memory fallback' })
    expect(describeBm25({ ...health, bm25_backend: 'elasticsearch' })).toMatchObject({ tone: 'ok', banner: null })
    const down = describeBm25({ ...health, bm25_backend: 'elasticsearch', bm25_available: false })
    expect(down).toMatchObject({ tone: 'warn', status: 'Elasticsearch unreachable' })
    expect(down.banner?.message).toMatch(/semantic \(FAISS\) retrieval only/)
  })
})
