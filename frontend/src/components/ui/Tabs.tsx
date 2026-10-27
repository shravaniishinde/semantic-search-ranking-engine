import type { ReactNode } from 'react'

export interface TabItem<K extends string> {
  key: K
  label: ReactNode
  count?: number
  dotClass?: string
}

export function Tabs<K extends string>({ items, value, onChange, ariaLabel }: { items: TabItem<K>[]; value: K; onChange: (key: K) => void; ariaLabel: string }) {
  return (
    <div role="tablist" aria-label={ariaLabel} className="flex gap-1 overflow-x-auto border-b border-line">
      {items.map((item) => {
        const active = item.key === value
        return (
          <button
            key={item.key}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(item.key)}
            className={`-mb-px flex items-center gap-2 whitespace-nowrap border-b-2 px-3 py-2 text-[13px] font-medium transition-colors ${
              active ? 'border-accent text-fg' : 'border-transparent text-muted hover:text-fg'
            }`}
          >
            {item.dotClass && <span aria-hidden className={`h-2 w-2 rounded-full ${item.dotClass}`} />}
            {item.label}
            {item.count !== undefined && <span className="num rounded bg-surface-2 px-1.5 text-[11px] text-muted">{item.count}</span>}
          </button>
        )
      })}
    </div>
  )
}

export function SegmentedControl<K extends string>({ options, value, onChange, ariaLabel }: { options: { key: K; label: string }[]; value: K; onChange: (k: K) => void; ariaLabel: string }) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded-md border border-line bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.key}
          role="radio"
          aria-checked={o.key === value}
          onClick={() => onChange(o.key)}
          className={`rounded px-2 py-1 text-xs font-medium ${o.key === value ? 'bg-surface text-fg shadow-sm' : 'text-muted hover:text-fg'}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
