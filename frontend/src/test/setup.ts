import '@testing-library/jest-dom/vitest'
import { cleanup, configure } from '@testing-library/react'
import { afterEach } from 'vitest'

// Recharts' ResponsiveContainer needs ResizeObserver, which jsdom does not provide.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver

// Lazy-loaded pages (Analytics, Clusters) can take >1s to import on a cold, busy test worker.
configure({ asyncUtilTimeout: 5000 })

// jsdom has no layout, so Recharts warns that charts are 0x0; that is expected in tests.
// A plain override (not vi.spyOn) so tests calling vi.resetAllMocks() cannot disable it.
const warn = console.warn.bind(console)
console.warn = (...args: unknown[]) => {
  if (typeof args[0] === 'string' && args[0].includes('of chart should be greater than 0')) return
  warn(...args)
}

afterEach(() => cleanup())
