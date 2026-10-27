import { useCallback, useMemo, useState, type ReactNode } from 'react'
import type { SearchResultItem } from '../types/api'
import { AppContext, initialTheme, type DrawerTarget, type SearchSession, type Theme } from './appState'

export function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [session, setSession] = useState<SearchSession | null>(null)
  const [drawer, setDrawer] = useState<DrawerTarget | null>(null)

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark'
      document.documentElement.classList.toggle('dark', next === 'dark')
      try {
        localStorage.setItem('theme', next)
      } catch {
        /* storage unavailable: theme just won't persist */
      }
      return next
    })
  }, [])

  const openDocument = useCallback((documentId: string, result?: SearchResultItem) => setDrawer({ documentId, result }), [])
  const closeDocument = useCallback(() => setDrawer(null), [])

  const value = useMemo(
    () => ({ theme, toggleTheme, session, setSession, drawer, openDocument, closeDocument }),
    [theme, toggleTheme, session, drawer, openDocument, closeDocument],
  )
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}
