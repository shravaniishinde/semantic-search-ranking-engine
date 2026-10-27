import type { Document } from '../../types/api'
import { Badge } from '../ui/Badge'

export function ClusterBadge({ clusterId }: { clusterId: number | null }) {
  if (clusterId === null) return <span className="text-xs text-faint">N/A</span>
  if (clusterId === -1) return <Badge tone="neutral" title="DBSCAN noise point">noise</Badge>
  return <Badge tone="accent">#{clusterId}</Badge>
}

/** Compact status flags; the distinctions mirror app/clustering/dbscan_analysis.py. */
export function DocumentStatusBadges({ doc }: { doc: Pick<Document, 'indexed' | 'is_duplicate' | 'duplicate_candidate' | 'is_outlier'> }) {
  return (
    <div className="flex flex-wrap gap-1">
      {doc.indexed ? <Badge tone="ok">indexed</Badge> : <Badge tone="bad">excluded</Badge>}
      {doc.is_duplicate ? (
        <Badge tone="bad" title="Cosine and token-Jaccard thresholds both passed">confirmed duplicate</Badge>
      ) : (
        doc.duplicate_candidate && <Badge tone="warn" title="High embedding similarity to another document">duplicate candidate</Badge>
      )}
      {doc.is_outlier && <Badge tone="info" title="DBSCAN noise point: no dense neighbourhood">outlier</Badge>}
    </div>
  )
}
