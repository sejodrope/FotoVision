import type { ModelInfo } from '../../types'
import { CheckCircle, Clock } from 'lucide-react'

interface Props {
  models: ModelInfo[]
}

const ARCHITECTURES: Record<string, { params: string; source: string }> = {
  mobilenet_v2: { params: '3.4M', source: 'Sandler et al., 2018' },
  resnet50: { params: '25.6M', source: 'He et al., 2016' },
  efficientnet_b0: { params: '5.3M', source: 'Tan & Le, 2019' },
  vit_b_16: { params: '86.6M', source: 'Dosovitskiy et al., 2021' },
}

export function ModelComparison({ models }: Props) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-100">
        <h3 className="font-semibold text-gray-900">Arquiteturas do TCC</h3>
        <p className="text-sm text-gray-500">Status dos modelos e informações de referência</p>
      </div>
      <div className="divide-y divide-gray-100">
        {models.map((m) => {
          const arch = ARCHITECTURES[m.id]
          return (
            <div key={m.id} className="px-6 py-4 flex items-start gap-4">
              <div className="mt-0.5">
                {m.calibrated ? (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                ) : (
                  <Clock className="w-5 h-5 text-amber-400" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-gray-900">{m.label}</span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      m.calibrated
                        ? 'bg-green-100 text-green-700'
                        : 'bg-amber-100 text-amber-700'
                    }`}
                  >
                    {m.calibrated ? 'Calibrado' : 'Aguardando treinamento'}
                  </span>
                </div>
                <p className="text-sm text-gray-500 mt-0.5">{m.description}</p>
                <p className="text-xs text-gray-400 mt-1">
                  {arch.params} parâmetros · {arch.source}
                </p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
