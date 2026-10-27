import { createContext, useContext } from 'react'
import type { SearchRequest, SearchResponse, SearchResultItem } from '../types/api'

// ------------------------------------------------------------------ theme
export type Theme = 'light' | 'dark'

export function initialTheme(): Theme {
  return typeof document !== 'undefined' && document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

// ------------------------------------------------------------------ search session
/** Last search survives page navigation, so Search → Analytics → Search keeps the results. */
export interface SearchSession {
  request: SearchRequest
  response: SearchResponse
}

// ------------------------------------------------------------------ document drawer
export interface DrawerTarget {
  documentId: string
  /** When opened from a search result, the scores/features of that result are shown too. */
  result?: SearchResultItem
}

export interface AppState {
  theme: Theme
  toggleTheme: () => void
  session: SearchSession | null
  setSession: (s: SearchSession | null) => void
  drawer: DrawerTarget | null
  openDocument: (documentId: string, result?: SearchResultItem) => void
  closeDocument: () => void
}

export const AppContext = createContext<AppState | null>(null)

export function useApp(): AppState {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used inside <AppProvider>')
  return ctx
}
