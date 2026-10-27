import { Outlet, useLocation } from 'react-router-dom'
import { DocumentDrawer } from '../documents/DocumentDrawer'
import { NAV_ITEMS } from './nav'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

export function AppLayout() {
  const { pathname } = useLocation()
  const item = NAV_ITEMS.find((n) => pathname.startsWith(n.path))
  return (
    <div className="flex h-full">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar item={item} />
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[1400px] px-4 py-5 lg:px-6">
            <Outlet />
          </div>
        </main>
      </div>
      <DocumentDrawer />
    </div>
  )
}
