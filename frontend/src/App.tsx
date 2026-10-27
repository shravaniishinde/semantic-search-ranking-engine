import { QueryClientProvider } from '@tanstack/react-query'
import { lazy, Suspense, useState } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/layout/AppLayout'
import { AppProvider } from './context/AppContext'
import { createQueryClient } from './services/queryClient'
import { DocumentsPage } from './pages/DocumentsPage'
import { IngestionPage } from './pages/IngestionPage'
import { SearchPage } from './pages/SearchPage'
import { SystemPage } from './pages/SystemPage'
import { LoadingState } from './components/ui/States'

// Chart-heavy pages are split out so the Search page does not download Recharts.
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage').then((m) => ({ default: m.AnalyticsPage })))
const ClustersPage = lazy(() => import('./pages/ClustersPage').then((m) => ({ default: m.ClustersPage })))

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to="/search" replace />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/clusters" element={<Suspense fallback={<LoadingState />}><ClustersPage /></Suspense>} />
        <Route path="/analytics" element={<Suspense fallback={<LoadingState />}><AnalyticsPage /></Suspense>} />
        <Route path="/ingestion" element={<IngestionPage />} />
        <Route path="/system" element={<SystemPage />} />
        <Route path="*" element={<Navigate to="/search" replace />} />
      </Route>
    </Routes>
  )
}

export default function App() {
  const [client] = useState(createQueryClient)
  return (
    <QueryClientProvider client={client}>
      <AppProvider>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </AppProvider>
    </QueryClientProvider>
  )
}
