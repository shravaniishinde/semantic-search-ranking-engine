import type { DocumentIn } from '../types/api'

export interface ParseResult {
  documents: DocumentIn[]
  errors: string[]
  format: 'json' | 'jsonl'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Mirrors app/ingestion/loader.py: JSONL (one object per line), a JSON list, or an object with a
 * "documents" list. Schema validation itself stays on the backend.
 */
export function parseDocuments(text: string, fileName = ''): ParseResult {
  const trimmed = text.trim()
  const looksLikeJsonl = fileName.toLowerCase().endsWith('.jsonl') || (!trimmed.startsWith('[') && trimmed.includes('\n'))
  if (!trimmed) return { documents: [], errors: ['The file is empty.'], format: looksLikeJsonl ? 'jsonl' : 'json' }

  if (looksLikeJsonl) {
    const documents: DocumentIn[] = []
    const errors: string[] = []
    text.split(/\r?\n/).forEach((line, i) => {
      if (!line.trim()) return
      try {
        const value: unknown = JSON.parse(line)
        if (isRecord(value)) documents.push(value as DocumentIn)
        else errors.push(`Line ${i + 1}: expected a JSON object`)
      } catch {
        errors.push(`Line ${i + 1}: invalid JSON`)
      }
    })
    return { documents, errors, format: 'jsonl' }
  }

  try {
    const value: unknown = JSON.parse(trimmed)
    const list = Array.isArray(value) ? value : isRecord(value) && Array.isArray(value.documents) ? value.documents : null
    if (!list) {
      if (isRecord(value)) return { documents: [value as DocumentIn], errors: [], format: 'json' }
      return { documents: [], errors: ['Expected a JSON list or an object with a "documents" list.'], format: 'json' }
    }
    const documents = list.filter(isRecord) as DocumentIn[]
    const skipped = list.length - documents.length
    return { documents, errors: skipped ? [`${skipped} entries are not JSON objects and were skipped.`] : [], format: 'json' }
  } catch (e) {
    return { documents: [], errors: [`Invalid JSON: ${(e as Error).message}`], format: 'json' }
  }
}

export function contentOf(doc: DocumentIn): string {
  const value = doc.content ?? doc.text ?? doc.body
  return typeof value === 'string' ? value : ''
}
