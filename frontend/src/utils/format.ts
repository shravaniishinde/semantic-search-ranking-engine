export const NA = 'N/A'

export function fmtNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return NA
  return value.toFixed(digits)
}

export function fmtMs(value: number | null | undefined): string {
  if (value === null || value === undefined) return NA
  if (value < 0.01) return '<0.01 ms'
  if (value < 1) return `${value.toFixed(2)} ms`
  if (value < 100) return `${value.toFixed(1)} ms`
  return `${Math.round(value)} ms`
}

export function fmtPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return NA
  return `${(value * 100).toFixed(digits)}%`
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined) return NA
  return value.toLocaleString('en-US')
}

/** "entity_lookup" -> "Entity lookup", "cloud computing" -> "Cloud computing" */
export function humanize(value: string): string {
  const s = value.replace(/[_-]+/g, ' ').trim()
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1).trimEnd()}…`
}

export function fmtDateTime(iso: string | undefined | null): string {
  if (!iso) return NA
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}
