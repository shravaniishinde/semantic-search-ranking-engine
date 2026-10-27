import { CheckCircle2, FileJson, Loader2, Upload, X } from 'lucide-react'
import { useRef, useState, type DragEvent } from 'react'
import { Link } from 'react-router-dom'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { ErrorState } from '../components/ui/States'
import { useIngest } from '../hooks/queries'
import type { IngestionReport } from '../types/api'
import { contentOf, parseDocuments, type ParseResult } from '../utils/ingestion'
import { fmtInt, truncate } from '../utils/format'

interface Selected {
  name: string
  size: number
  parsed: ParseResult
}

const MAX_PREVIEW = 8

function ReportView({ report }: { report: IngestionReport }) {
  const rows: [string, number, string][] = [
    ['Received', report.received, 'documents in this upload'],
    ['Accepted', report.accepted, 'passed validation and cleaning'],
    ['Rejected', report.rejected.length, 'see reasons below'],
  ]
  const totals: [string, number][] = [
    ['Indexed', report.indexed],
    ['Excluded', report.excluded],
    ['Confirmed duplicates', report.duplicates],
    ['Outliers', report.outliers],
  ]
  return (
    <div className="space-y-4" data-testid="ingestion-report">
      <div className="flex items-center gap-2 text-[13px] font-medium text-ok">
        <CheckCircle2 size={16} /> Ingestion finished. DBSCAN and both indexes were rebuilt over the whole corpus.
      </div>
      <div className="grid grid-cols-3 gap-2">
        {rows.map(([label, value, hint]) => (
          <div key={label} className="rounded-md border border-line px-3 py-2">
            <div className="text-[11px] text-faint">{label}</div>
            <div className="num text-xl font-semibold">{value}</div>
            <div className="text-[11px] text-faint">{hint}</div>
          </div>
        ))}
      </div>
      <div>
        <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-faint">Corpus totals after this ingest</div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {totals.map(([label, value]) => (
            <div key={label} className="rounded-md bg-surface-2 px-3 py-2">
              <div className="text-[11px] text-faint">{label}</div>
              <div className="num font-semibold">{fmtInt(value)}</div>
            </div>
          ))}
        </div>
      </div>
      {report.rejected.length > 0 && (
        <div>
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-faint">Rejected documents</div>
          <ul className="space-y-1 text-[13px]">
            {report.rejected.map((r, i) => (
              <li key={`${r.document_id}-${i}`} className="flex gap-2">
                <span className="font-mono text-xs text-fg">{r.document_id ?? '(no id)'}</span>
                <span className="text-muted">{r.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="flex gap-2 text-[13px]">
        <Link to="/documents" className="text-accent hover:underline">
          Browse documents
        </Link>
        <span className="text-faint">·</span>
        <Link to="/clusters" className="text-accent hover:underline">
          Review clusters &amp; duplicates
        </Link>
      </div>
    </div>
  )
}

export function IngestionPage() {
  const [selected, setSelected] = useState<Selected | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const ingest = useIngest()

  const readFile = async (file: File) => {
    ingest.reset()
    const text = await file.text()
    setSelected({ name: file.name, size: file.size, parsed: parseDocuments(text, file.name) })
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) void readFile(file)
  }

  const docs = selected?.parsed.documents ?? []
  const missingContent = docs.filter((d) => !contentOf(d).trim()).length

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_420px]">
      <div className="space-y-4">
        <Card title="1 · Select a file" subtitle="JSON list, {&quot;documents&quot;: [...]} object, or JSONL (one document per line)">
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`flex flex-col items-center justify-center rounded-md border border-dashed px-6 py-10 text-center ${dragging ? 'border-accent bg-accent-soft' : 'border-line-strong bg-bg'}`}
          >
            <Upload size={24} className="text-faint" />
            <p className="mt-2 text-[13px] text-fg">Drop a .json or .jsonl file here</p>
            <p className="mt-0.5 text-xs text-muted">Fields: document_id, title, content (or text), source, category, created_at, metadata</p>
            <Button className="mt-3" onClick={() => inputRef.current?.click()}>
              Choose file
            </Button>
            <input
              ref={inputRef}
              type="file"
              accept=".json,.jsonl,application/json"
              aria-label="Document file"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) void readFile(file)
                e.target.value = ''
              }}
            />
          </div>
        </Card>

        {selected && (
          <Card
            title="2 · Preview"
            subtitle={`${selected.name} · ${(selected.size / 1024).toFixed(1)} KB · ${selected.parsed.format.toUpperCase()}`}
            actions={
              <button aria-label="Remove file" onClick={() => setSelected(null)} className="rounded p-1 text-muted hover:bg-surface-2">
                <X size={15} />
              </button>
            }
          >
            <div className="mb-3 flex flex-wrap gap-2 text-xs">
              <Badge tone="accent">{docs.length} documents parsed</Badge>
              {selected.parsed.errors.length > 0 && <Badge tone="bad">{selected.parsed.errors.length} parse errors</Badge>}
              {missingContent > 0 && <Badge tone="warn">{missingContent} without content (backend will reject)</Badge>}
            </div>
            {selected.parsed.errors.length > 0 && (
              <ul className="mb-3 max-h-28 overflow-y-auto rounded-md bg-bad-soft px-3 py-2 text-xs text-bad">
                {selected.parsed.errors.slice(0, 20).map((e) => (
                  <li key={e}>{e}</li>
                ))}
              </ul>
            )}
            {docs.length > 0 && (
              <div className="overflow-x-auto rounded-md border border-line">
                <table className="w-full text-[13px]">
                  <thead className="border-b border-line text-left text-[11px] uppercase tracking-wide text-faint">
                    <tr>
                      <th className="px-3 py-2 font-semibold">ID</th>
                      <th className="px-3 py-2 font-semibold">Title</th>
                      <th className="px-3 py-2 font-semibold">Category</th>
                      <th className="px-3 py-2 text-right font-semibold">Chars</th>
                    </tr>
                  </thead>
                  <tbody>
                    {docs.slice(0, MAX_PREVIEW).map((d, i) => (
                      <tr key={i} className="border-b border-line/60 last:border-0">
                        <td className="px-3 py-1.5 font-mono text-xs text-muted">{String(d.document_id ?? d.id ?? '(auto)')}</td>
                        <td className="px-3 py-1.5">{truncate(String(d.title ?? ''), 60) || <span className="text-faint">(untitled)</span>}</td>
                        <td className="px-3 py-1.5 text-muted">{String(d.category ?? 'uncategorized')}</td>
                        <td className="num px-3 py-1.5 text-right text-muted">{contentOf(d).length}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {docs.length > MAX_PREVIEW && <div className="border-t border-line px-3 py-1.5 text-xs text-faint">…and {docs.length - MAX_PREVIEW} more</div>}
              </div>
            )}
            <div className="mt-4 flex items-center justify-end gap-3">
              <span className="text-xs text-muted">Existing IDs are updated in place.</span>
              <Button variant="primary" disabled={docs.length === 0 || ingest.isPending} onClick={() => ingest.mutate(docs)}>
                {ingest.isPending ? <Loader2 size={14} className="animate-spin" /> : <FileJson size={14} />}
                {ingest.isPending ? 'Ingesting…' : `Ingest ${docs.length} documents`}
              </Button>
            </div>
          </Card>
        )}
      </div>

      <Card title="3 · Result">
        {ingest.isPending && (
          <div role="status" className="flex items-center gap-2 py-6 text-[13px] text-muted">
            <Loader2 size={15} className="animate-spin" /> Cleaning, NER, embedding, DBSCAN analysis and re-indexing…
          </div>
        )}
        {ingest.error && <ErrorState error={ingest.error} />}
        {ingest.data && <ReportView report={ingest.data} />}
        {!ingest.isPending && !ingest.error && !ingest.data && (
          <p className="py-6 text-[13px] text-muted">
            The ingestion report from <code className="font-mono text-xs">POST /documents/ingest</code> appears here. Documents are cleaned, run through NER and embedded,
            then DBSCAN re-analyses the corpus and confirmed duplicates are excluded from the index.
          </p>
        )}
      </Card>
    </div>
  )
}
