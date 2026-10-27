import { Moon, Sun } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useApp } from '../../context/appState'
import { useHealth } from '../../hooks/queries'
import { fmtMs } from '../../utils/format'
import { StatusDot } from '../ui/Badge'
import type { NavItem } from './nav'

export function BackendStatus() {
  const health = useHealth()
  let tone: 'ok' | 'bad' | 'neutral' = 'neutral'
  let label = 'Checking API…'
  if (health.data) {
    tone = 'ok'
    label = `API online · ${fmtMs(health.data.roundTripMs)}`
  } else if (health.error) {
    tone = 'bad'
    label = 'API unreachable'
  }
  const backend = health.data?.data.bm25_backend
  return (
    <Link to="/system" className="flex items-center gap-2 rounded-md border border-line px-2.5 py-1 text-xs text-muted hover:bg-surface-2" title="Round trip of the last /health call, measured in the browser">
      <StatusDot tone={tone} />
      <span className="num">{label}</span>
      {backend && <span className="hidden border-l border-line pl-2 sm:inline">BM25: {backend === 'memory' ? 'in-memory' : backend}</span>}
    </Link>
  )
}

export function TopBar({ item }: { item?: NavItem }) {
  const { theme, toggleTheme } = useApp()
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-line bg-surface px-4 lg:px-6">
      <div className="min-w-0">
        <h1 className="truncate text-[15px] font-semibold tracking-tight">{item?.title ?? 'Semantic Search Engine'}</h1>
        {item && <p className="hidden truncate text-xs text-muted md:block">{item.description}</p>}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <BackendStatus />
        <button
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          className="rounded-md border border-line p-1.5 text-muted hover:bg-surface-2 hover:text-fg"
        >
          {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>
    </header>
  )
}
