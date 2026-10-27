import type { Entity } from '../../types/api'

const LABEL_STYLE: Record<string, string> = {
  ORG: 'text-info',
  PRODUCT: 'text-hybrid',
  PERSON: 'text-rerank',
  DATE: 'text-muted',
  EVENT: 'text-bm25',
  GPE: 'text-semantic',
  VULNERABILITY_ID: 'text-bad',
}

export function EntityChips({ entities, max, empty = 'No entities detected' }: { entities: Entity[]; max?: number; empty?: string }) {
  if (entities.length === 0) return <span className="text-xs text-faint">{empty}</span>
  const shown = max ? entities.slice(0, max) : entities
  return (
    <div className="flex flex-wrap gap-1.5">
      {shown.map((e) => (
        <span key={`${e.text}-${e.label}`} className="inline-flex items-center gap-1.5 rounded border border-line bg-surface-2 px-1.5 py-0.5 text-xs">
          <span className="text-fg">{e.text}</span>
          <span className={`font-mono text-[10px] font-semibold ${LABEL_STYLE[e.label] ?? 'text-faint'}`}>{e.label}</span>
        </span>
      ))}
      {max && entities.length > max && <span className="self-center text-xs text-faint">+{entities.length - max} more</span>}
    </div>
  )
}
