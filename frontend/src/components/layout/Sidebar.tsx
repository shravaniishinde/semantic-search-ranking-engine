import { Search } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { NAV_ITEMS } from './nav'

/** Full sidebar from lg; an icon rail below that so navigation stays reachable on tablets. */
export function Sidebar() {
  return (
    <nav aria-label="Main" className="flex h-full w-14 shrink-0 flex-col border-r border-line bg-surface lg:w-56">
      <div className="flex h-14 items-center gap-2.5 border-b border-line px-3 lg:px-4">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-accent text-white dark:text-[#0d1016]">
          <Search size={15} strokeWidth={2.5} />
        </div>
        <div className="hidden min-w-0 lg:block">
          <div className="truncate text-[13px] font-semibold leading-tight">Semantic Search Engine</div>
          <div className="truncate text-[11px] text-faint">Hybrid retrieval &amp; ranking</div>
        </div>
      </div>
      <ul className="flex flex-1 flex-col gap-0.5 p-2">
        {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
          <li key={path}>
            <NavLink
              to={path}
              title={label}
              className={({ isActive }) =>
                `flex items-center justify-center gap-2.5 rounded-md px-2 py-2 text-[13px] font-medium lg:justify-start lg:px-2.5 ${
                  isActive ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-surface-2 hover:text-fg'
                }`
              }
            >
              <Icon size={16} aria-hidden />
              <span className="hidden lg:inline">{label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="hidden border-t border-line px-4 py-3 text-[11px] leading-relaxed text-faint lg:block">
        MiniLM · FAISS · BM25
        <br />
        XGBoost · DBSCAN · spaCy
      </div>
    </nav>
  )
}
