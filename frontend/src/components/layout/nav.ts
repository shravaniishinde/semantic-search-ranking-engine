import { Activity, BarChart3, Boxes, FileText, Search, Upload, type LucideIcon } from 'lucide-react'

export interface NavItem {
  path: string
  label: string
  title: string
  description: string
  icon: LucideIcon
}

export const NAV_ITEMS: NavItem[] = [
  { path: '/search', label: 'Search', title: 'Hybrid Search', description: 'BM25 + FAISS retrieval, RRF fusion and XGBoost reranking', icon: Search },
  { path: '/documents', label: 'Documents', title: 'Document Explorer', description: 'Browse the corpus with NER, cluster and duplicate annotations', icon: FileText },
  { path: '/clusters', label: 'Clusters', title: 'Cluster Analysis', description: 'DBSCAN clusters, near-duplicates and outliers over document embeddings', icon: Boxes },
  { path: '/analytics', label: 'Analytics', title: 'Retrieval Analytics', description: 'Offline evaluation, corpus statistics and live search metrics', icon: BarChart3 },
  { path: '/ingestion', label: 'Ingestion', title: 'Document Ingestion', description: 'Upload JSON / JSONL documents into the index', icon: Upload },
  { path: '/system', label: 'System', title: 'System Status', description: 'Backend health, retrieval backends and model status', icon: Activity },
]
