import { Loader2, Search, X } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { humanize } from '../../utils/format'
import { Button, inputClass } from '../ui/Button'

export interface SearchControls {
  query: string
  topK: number
  category: string
  source: string
}

interface SearchBarProps {
  initial: SearchControls
  categories: string[]
  sources: string[]
  loading: boolean
  onSearch: (controls: SearchControls) => void
  /** Called when a filter changes; the page decides whether to re-run the active query. */
  onControlsChange: (controls: SearchControls) => void
  onClear: () => void
}

const TOP_K_OPTIONS = [5, 10, 20, 30]

export function SearchBar({ initial, categories, sources, loading, onSearch, onControlsChange, onClear }: SearchBarProps) {
  const [controls, setControls] = useState<SearchControls>(initial)

  const update = (patch: Partial<SearchControls>, notify = true) => {
    const next = { ...controls, ...patch }
    setControls(next)
    if (notify) onControlsChange(next)
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (controls.query.trim()) onSearch(controls)
  }

  return (
    <form onSubmit={submit} role="search" className="rounded-lg border border-line bg-surface p-3">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-faint" aria-hidden />
          <input
            aria-label="Search query"
            value={controls.query}
            onChange={(e) => update({ query: e.target.value }, false)}
            placeholder="Search documents… e.g. how do I stop brute-force login attempts?"
            maxLength={512}
            className="h-11 w-full rounded-md border border-line bg-bg pl-9 pr-9 text-[15px] text-fg placeholder:text-faint focus:border-accent focus:outline-none"
          />
          {controls.query && (
            <button
              type="button"
              aria-label="Clear search"
              onClick={() => {
                setControls({ ...controls, query: '' })
                onClear()
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-faint hover:text-fg"
            >
              <X size={15} />
            </button>
          )}
        </div>
        {/* Not disabled while loading: a disabled default button blocks Enter-to-submit, so a new
            query typed during an in-flight search would be silently ignored. The latest
            submission wins (TanStack Query only runs the last mutate call's callbacks). */}
        <Button type="submit" variant="primary" disabled={!controls.query.trim()} className="h-11 px-5">
          {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
          Search
        </Button>
      </div>
      <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted">
        <label className="flex items-center gap-1.5">
          Top-K
          <select aria-label="Top-K" value={controls.topK} onChange={(e) => update({ topK: Number(e.target.value) })} className={inputClass}>
            {TOP_K_OPTIONS.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          Category
          <select aria-label="Category filter" value={controls.category} onChange={(e) => update({ category: e.target.value })} className={inputClass}>
            <option value="">All categories</option>
            {categories.map((c) => (
              <option key={c} value={c}>
                {humanize(c)}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          Source
          <select aria-label="Source filter" value={controls.source} onChange={(e) => update({ source: e.target.value })} className={inputClass}>
            <option value="">All sources</option>
            {sources.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <span className="ml-auto hidden text-faint md:inline">Press Enter to search · filters re-run the current query</span>
      </div>
    </form>
  )
}
