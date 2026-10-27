import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

interface StatCardProps {
  label: string
  value: ReactNode
  hint?: ReactNode
  icon?: LucideIcon
  loading?: boolean
}

export function StatCard({ label, value, hint, icon: Icon, loading }: StatCardProps) {
  return (
    <div className="rounded-lg border border-line bg-surface px-4 py-3">
      <div className="flex items-center justify-between text-xs text-muted">
        <span>{label}</span>
        {Icon && <Icon size={14} className="text-faint" aria-hidden />}
      </div>
      <div className="num mt-1 text-2xl font-semibold tracking-tight text-fg">
        {loading ? <span className="inline-block h-7 w-12 animate-pulse rounded bg-surface-2" /> : value}
      </div>
      {hint && <div className="mt-0.5 text-xs text-faint">{hint}</div>}
    </div>
  )
}

export function StatGrid({ children }: { children: ReactNode }) {
  return <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">{children}</div>
}
