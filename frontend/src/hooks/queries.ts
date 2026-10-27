// React Query wrappers: caching + de-duplication so pages don't refetch on every render.
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import * as api from '../services/api'
import type { DocumentIn, DocumentQuery } from '../types/api'

export const queryKeys = {
  health: ['health'] as const,
  analytics: ['analytics'] as const,
  clusters: ['clusters'] as const,
  evaluation: ['evaluation'] as const,
  documents: (q: DocumentQuery) => ['documents', q] as const,
  document: (id: string) => ['document', id] as const,
}

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: api.getHealth,
    refetchInterval: 30_000,
    staleTime: 10_000,
    retry: false,
  })
}

export function useAnalytics() {
  return useQuery({ queryKey: queryKeys.analytics, queryFn: api.getAnalytics, staleTime: 30_000 })
}

export function useClusters() {
  return useQuery({ queryKey: queryKeys.clusters, queryFn: api.getClusters, staleTime: 60_000 })
}

export function useEvaluation() {
  return useQuery({ queryKey: queryKeys.evaluation, queryFn: api.getEvaluation, staleTime: 5 * 60_000 })
}

export function useDocuments(query: DocumentQuery, enabled = true) {
  return useQuery({
    queryKey: queryKeys.documents(query),
    queryFn: () => api.getDocuments(query),
    placeholderData: keepPreviousData,
    staleTime: 30_000,
    enabled,
  })
}

export function useDocument(id: string | null) {
  return useQuery({
    queryKey: queryKeys.document(id ?? ''),
    queryFn: () => api.getDocument(id as string),
    enabled: !!id,
    staleTime: 60_000,
  })
}

export function useSearch() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: api.searchDocuments,
    // Search stats on the Analytics page change after every search.
    onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.analytics }),
  })
}

/** Ingestion re-runs DBSCAN and re-indexes, so every corpus-derived view is stale afterwards. */
export function useIngest() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (docs: DocumentIn[]) => api.ingestDocuments(docs),
    onSuccess: () =>
      Promise.all(
        [queryKeys.analytics, queryKeys.clusters, queryKeys.health, ['documents'], ['document']].map((queryKey) =>
          client.invalidateQueries({ queryKey }),
        ),
      ),
  })
}

export function useRunEvaluation() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: api.runEvaluation,
    onSuccess: (data) => client.setQueryData(queryKeys.evaluation, data),
  })
}
