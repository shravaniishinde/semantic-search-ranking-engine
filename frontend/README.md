# Dashboard (frontend)

React 19 + TypeScript + Vite + Tailwind CSS 4 + Recharts + TanStack Query. Consumes the FastAPI
backend in the parent directory; see the main [README](../README.md#dashboard) for the full guide.

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # vitest
npm run build      # type-check + production build into dist/
```

`VITE_API_BASE_URL` (see `.env.example`) sets the backend address; it defaults to
`http://127.0.0.1:8000`.

```
src/
  services/api.ts      single API client (fetch wrapper + ApiError)
  hooks/queries.ts     TanStack Query hooks (caching, invalidation after ingest)
  types/api.ts         TypeScript mirrors of the backend response models
  context/             theme, persisted search session, document drawer
  components/          layout, ui primitives, search, documents, charts
  pages/               Search, Documents, Clusters, Analytics, Ingestion, System
  utils/               formatting, ranking-feature display metadata, ingestion parsing
  test/                vitest + Testing Library tests and fixtures
```
