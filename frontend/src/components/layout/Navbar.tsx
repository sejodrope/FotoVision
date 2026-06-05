import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Leaf, Clock, BarChart2, Microscope, FlaskConical } from 'lucide-react'
import clsx from 'clsx'
import { ModelInfoModal } from './ModelInfoModal'

const links = [
  { to: '/', label: 'Diagnóstico', icon: Microscope, end: true },
  { to: '/dashboard', label: 'Pesquisa', icon: BarChart2, end: false },
  { to: '/history', label: 'Histórico', icon: Clock, end: false },
]

export function Navbar() {
  const [showModel, setShowModel] = useState(false)

  return (
    <>
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-2">
              <Leaf className="text-primary-600 w-7 h-7" />
              <span className="font-bold text-xl text-gray-900">FitoVision</span>
            </div>
            <div className="flex items-center gap-1">
              {links.map(({ to, label, icon: Icon, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  className={({ isActive }) =>
                    clsx(
                      'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-primary-50 text-primary-700'
                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900',
                    )
                  }
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:block">{label}</span>
                </NavLink>
              ))}

              <div className="w-px h-5 bg-gray-200 mx-1" />

              <button
                onClick={() => setShowModel(true)}
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors"
                title="Métricas do modelo"
              >
                <FlaskConical className="w-4 h-4" />
                <span className="hidden sm:block">Modelo</span>
              </button>
            </div>
          </div>
        </div>
      </nav>

      {showModel && <ModelInfoModal onClose={() => setShowModel(false)} />}
    </>
  )
}