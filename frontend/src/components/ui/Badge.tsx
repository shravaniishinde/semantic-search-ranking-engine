import type { ReactNode } from 'react'

export type Tone = 'neutral' | 'accent' | 'ok' | 'warn' | 'bad' | 'info' | 'bm25' | 'semantic' | 'hybrid' | 'rerank'

const TONES: Record<Tone, string> = {
  neutral: 'bg-surface-2 text-muted border-line',
  accent: 'bg-accent-soft text-accent border-transparent',
  ok: 'bg-ok-soft text-ok border-transparent',
  warn: 'bg-warn-soft text-warn border-transparent',
  bad: 'bg-bad-soft text-bad border-transparent',
  info: 'bg-info-soft text-info border-transparent',
  bm25: 'text-bm25 border-bm25/40 bg-bm25/10',
  semantic: 'text-semantic border-semantic/40 bg-semantic/10',
  hybrid: 'text-hybrid border-hybrid/40 bg-hybrid/10',
  rerank: 'text-rerank border-rerank/40 bg-rerank/10',
}

export function Badge({ children, tone = 'neutral', title, className = '' }: { children: ReactNode; tone?: Tone; title?: string; className?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 whitespace-nowrap rounded border px-1.5 py-0.5 text-[11px] font-medium leading-4 ${TONES[tone]} ${className}`}
    >
      {children}
    </span>
  )
}

export function StatusDot({ tone }: { tone: 'ok' | 'warn' | 'bad' | 'info' | 'neutral' }) {
  const color = { ok: 'bg-ok', warn: 'bg-warn', bad: 'bg-bad', info: 'bg-info', neutral: 'bg-faint' }[tone]
  return <span aria-hidden className={`inline-block h-2 w-2 shrink-0 rounded-full ${color}`} />
}
