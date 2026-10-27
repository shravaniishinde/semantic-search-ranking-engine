import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { analytics, clusters, evaluation, health, searchResponse } from './fixtures'
import { renderApp } from './render'

vi.mock('../services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../services/api')>()
  return {
    ...actual,
    getHealth: vi.fn(),
    getAnalytics: vi.fn(),
    getClusters: vi.fn(),
    getEvaluation: vi.fn(),
    runEvaluation: vi.fn(),
    getDocuments: vi.fn(),
    getDocument: vi.fn(),
    searchDocuments: vi.fn(),
    ingestDocuments: vi.fn(),
  }
})

const mocked = vi.mocked(api)

beforeEach(() => {
  vi.resetAllMocks()
  mocked.getHealth.mockResolvedValue({ data: health, roundTripMs: 4.2 })
  mocked.getAnalytics.mockResolvedValue(analytics)
  mocked.getClusters.mockResolvedValue(clusters)
  mocked.getEvaluation.mockResolvedValue(evaluation)
  mocked.getDocuments.mockResolvedValue({ total: 0, offset: 0, limit: 20, documents: [] })
})

async function runSearch(query = 'security architecture for AWS') {
  const user = userEvent.setup()
  renderApp('/search')
  await user.type(screen.getByRole('textbox', { name: 'Search query' }), `${query}{Enter}`)
  return user
}

describe('Search page', () => {
  it('shows the empty state before any search', () => {
    renderApp('/search')
    expect(screen.getByText('Search the document collection to see ranked results.')).toBeInTheDocument()
    expect(mocked.searchDocuments).not.toHaveBeenCalled()
  })

  it('submits on Enter with the selected controls and shows a loading state', async () => {
    let resolve!: (r: typeof searchResponse) => void
    mocked.searchDocuments.mockReturnValue(new Promise((r) => (resolve = r)))
    await runSearch()
    expect(mocked.searchDocuments).toHaveBeenCalledWith(
      { query: 'security architecture for AWS', top_k: 10, filters: null, include_retrieval: true },
      expect.anything(),
    )
    expect(screen.getByRole('status')).toHaveTextContent(/Running NER/)
    resolve(searchResponse)
    expect(await screen.findByTestId('intent-label')).toHaveTextContent('Technical')
  })

  it('accepts a new query while a previous search is still in flight', async () => {
    let resolveFirst!: (r: typeof searchResponse) => void
    mocked.searchDocuments
      .mockReturnValueOnce(new Promise((r) => (resolveFirst = r)))
      .mockResolvedValueOnce({ ...searchResponse, query: 'second query' })
    const user = await runSearch('first query')
    const box = screen.getByRole('textbox', { name: 'Search query' })
    await user.clear(box)
    await user.type(box, 'second query{Enter}')
    expect(mocked.searchDocuments).toHaveBeenCalledTimes(2)
    expect(mocked.searchDocuments.mock.calls[1][0]).toMatchObject({ query: 'second query' })
    resolveFirst(searchResponse) // the stale response must not replace the newer one
    expect(await screen.findByText('second query', { selector: 'code' })).toBeInTheDocument()
  })

  it('renders query insights, pipeline and final results from the API response', async () => {
    mocked.searchDocuments.mockResolvedValue(searchResponse)
    await runSearch()
    expect(await screen.findByTestId('intent-confidence')).toHaveTextContent('98.3% confidence')
    const insights = screen.getByRole('region', { name: 'Query insights' })
    expect(within(insights).getByText('AWS')).toBeInTheDocument()
    expect(screen.getByTestId('total-latency')).toHaveTextContent('62.7 ms')

    const pipeline = screen.getByRole('region', { name: 'Retrieval pipeline' })
    expect(within(pipeline).getByText('21 candidates')).toBeInTheDocument() // BM25 count from retrieval_counts
    expect(within(pipeline).getByText('30 merged')).toBeInTheDocument()

    const cards = screen.getAllByRole('article')
    expect(cards).toHaveLength(2)
    expect(within(cards[0]).getByText('Designing a Secure Architecture on AWS')).toBeInTheDocument()
    expect(within(cards[0]).getByText('3.876')).toBeInTheDocument()
    // Second result was not retrieved by BM25: shown as a dash, not a fabricated 0.
    expect(within(cards[1]).getByText('—')).toBeInTheDocument()
  })

  it('expands the ranking explanation with backend features', async () => {
    mocked.searchDocuments.mockResolvedValue(searchResponse)
    const user = await runSearch()
    const second = (await screen.findAllByRole('article'))[1]
    const toggle = within(second).getByRole('button', { name: /Why this result/ })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await user.click(toggle)
    const explanation = within(second).getByTestId('ranking-explanation')
    expect(within(explanation).getByText('BM25 rank').closest('tr')).toHaveTextContent('not retrieved')
    expect(within(explanation).getByText('Entity overlap').closest('tr')).toHaveTextContent('100%')
    expect(within(explanation).getByText('Category match').closest('tr')).toHaveTextContent('Yes')
    expect(explanation).toHaveTextContent('Hybrid #1 → final #2')
  })

  it('switches between BM25, semantic and hybrid lists', async () => {
    mocked.searchDocuments.mockResolvedValue(searchResponse)
    const user = await runSearch()
    await screen.findAllByRole('article')
    await user.click(screen.getByRole('tab', { name: /BM25/ }))
    expect(screen.getByText('Serverless Computing with AWS Lambda')).toBeInTheDocument()
    expect(screen.getByText('25.542')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: /Semantic/ }))
    expect(screen.getByText('0.470')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: /Hybrid/ }))
    expect(screen.getByText('0.0161')).toBeInTheDocument()
  })

  it('shows a connection error when the backend is down', async () => {
    mocked.searchDocuments.mockRejectedValue(new api.ApiError('Unable to connect to the search API at http://localhost:8000.', null, 'network'))
    await runSearch()
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Unable to connect to the search API')
    expect(within(alert).getByRole('button', { name: /Retry/ })).toBeInTheDocument()
  })
})

describe('Navigation and pages', () => {
  it('navigates between pages from the sidebar', async () => {
    const user = userEvent.setup()
    renderApp('/search')
    await user.click(screen.getByRole('link', { name: 'Analytics' }))
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Retrieval Analytics')
    await user.click(screen.getByRole('link', { name: 'System' }))
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('System Status')
  })

  it('renders analytics KPIs and evaluation metrics from the API', async () => {
    renderApp('/analytics')
    const table = await screen.findByRole('table', { name: 'Evaluation metrics' })
    const rerankRow = within(table).getByText('Hybrid + reranker').closest('tr') as HTMLElement
    expect(rerankRow).toHaveTextContent('0.934')
    expect(within(table).getByText('Semantic (FAISS)').closest('tr')).toHaveTextContent('0.950')
    expect(screen.getByText('76')).toBeInTheDocument() // document_count KPI
    expect(screen.getByText(/20 held-out queries/)).toBeInTheDocument()
  })

  it('offers to run evaluation instead of inventing metrics when no report exists', async () => {
    mocked.getEvaluation.mockResolvedValue({ available: false, report: null })
    renderApp('/analytics')
    expect(await screen.findByText('No evaluation report yet')).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: 'Evaluation metrics' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Run evaluation/ })).toBeInTheDocument()
  })

  it('explains the in-memory BM25 fallback on the System page', async () => {
    renderApp('/system')
    expect(await screen.findByText('Retrieval backend: in-memory fallback')).toBeInTheDocument()
    expect(screen.getByText(/Elasticsearch is unavailable/)).toBeInTheDocument()
  })

  it('shows confirmed duplicates on the Clusters page', async () => {
    const user = userEvent.setup()
    renderApp('/clusters')
    await user.click(await screen.findByRole('tab', { name: /Duplicates/ }))
    expect(screen.getByText('confirmed duplicate')).toBeInTheDocument()
    expect(screen.getByText('0.9991')).toBeInTheDocument()
  })

  it('filters documents through the API', async () => {
    const user = userEvent.setup()
    renderApp('/documents')
    await waitFor(() => expect(mocked.getDocuments).toHaveBeenCalled())
    await user.selectOptions(screen.getByRole('combobox', { name: 'Status' }), 'outlier')
    await waitFor(() => expect(mocked.getDocuments).toHaveBeenLastCalledWith(expect.objectContaining({ is_outlier: true, offset: 0 })))
    expect(await screen.findByText('No documents match these filters.')).toBeInTheDocument()
  })
})
