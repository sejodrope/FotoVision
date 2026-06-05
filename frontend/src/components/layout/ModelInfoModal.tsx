import { useEffect } from 'react'
import { X, Cpu, Target, Zap, Database, CheckCircle } from 'lucide-react'

interface Props {
  onClose: () => void
}

const METRICS = [
  { icon: Target,   label: 'Acurácia — test set', value: '98,81%',  accent: '#16a34a' },
  { icon: Target,   label: 'F1 Score (macro)',     value: '98,67%',  accent: '#16a34a' },
  { icon: Zap,      label: 'Latência média',       value: '7,9 ms/img', accent: '#3b82f6' },
  { icon: Database, label: 'Dataset total',        value: '210.832 imgs', accent: '#8b5cf6' },
]

const CLASSES = [
  { label: 'Saudável',  count: '71.000',  color: '#16a34a', pct: '34%' },
  { label: 'Anômala',   count: '139.832', color: '#dc2626', pct: '66%' },
]

export function ModelInfoModal({ onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Informações do modelo"
    >
      {/* backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* panel */}
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md overflow-hidden">
        {/* header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-primary-50 flex items-center justify-center">
              <Cpu className="w-5 h-5 text-primary-600" />
            </div>
            <div>
              <h2 className="font-bold text-gray-900 leading-none">MobileNetV2</h2>
              <p className="text-xs text-gray-500 mt-0.5">Classificação binária — FitoVision TCC</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors"
            aria-label="Fechar"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* metrics grid */}
          <div className="grid grid-cols-2 gap-3">
            {METRICS.map(({ icon: Icon, label, value, accent }) => (
              <div key={label} className="bg-gray-50 rounded-xl p-4">
                <Icon className="w-4 h-4 mb-2" style={{ color: accent }} />
                <p className="text-xs text-gray-500 leading-snug">{label}</p>
                <p className="text-lg font-bold text-gray-900 mt-0.5" style={{ color: accent }}>
                  {value}
                </p>
              </div>
            ))}
          </div>

          {/* dataset breakdown */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
              Distribuição do dataset
            </p>
            <div className="space-y-2.5">
              {CLASSES.map(({ label, count, color, pct }) => (
                <div key={label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-gray-700">{label}</span>
                    <span className="text-gray-500 tabular-nums">{count} imgs ({pct})</span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: pct, backgroundColor: color }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* architecture note */}
          <div className="bg-primary-50 rounded-xl px-4 py-3 flex items-start gap-3">
            <CheckCircle className="w-4 h-4 text-primary-600 shrink-0 mt-0.5" />
            <p className="text-xs text-primary-800 leading-relaxed">
              Pesos treinados em 210k imagens de hortaliças folhosas provenientes de 14 datasets
              públicos. Transfer learning a partir de ImageNet. Avaliado em test set balanceado
              e isolado durante o treino.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}