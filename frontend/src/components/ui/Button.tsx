import type { ButtonHTMLAttributes } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost'

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-accent text-white hover:opacity-90 disabled:opacity-50 dark:text-[#0d1016]',
  secondary: 'border border-line bg-surface text-fg hover:bg-surface-2 disabled:opacity-50',
  ghost: 'text-muted hover:bg-surface-2 hover:text-fg disabled:opacity-50',
}

export function Button({ variant = 'secondary', className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...props}
      className={`inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors disabled:cursor-not-allowed ${VARIANTS[variant]} ${className}`}
    />
  )
}

export const inputClass =
  'rounded-md border border-line bg-surface px-2.5 py-1.5 text-[13px] text-fg placeholder:text-faint focus:border-accent focus:outline-none'
