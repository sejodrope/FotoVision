import { useState } from 'react'
import { Info } from 'lucide-react'

interface Props {
  original: string
  gradcam: string
  modelLabel: string
}

export function GradCamViewer({ original, gradcam, modelLabel }: Props) {
  const [showOverlay, setShowOverlay] = useState(true)

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-gray-800">Grad-CAM — {modelLabel}</h3>
          <div className="group relative">
            <Info className="w-4 h-4 text-gray-400 cursor-help" />
            <div className="hidden group-hover:block absolute left-6 top-0 z-10 w-64 bg-gray-900 text-white text-xs rounded-lg p-2 shadow-lg">
              Regiões em vermelho indicam onde o modelo focou para fazer o diagnóstico. Regiões em
              azul tiveram menos influência.
            </div>
          </div>
        </div>
        <div className="flex gap-2 text-sm">
          <button
            onClick={() => setShowOverlay(false)}
            className={`px-3 py-1 rounded-lg ${!showOverlay ? 'bg-gray-200 font-medium' : 'text-gray-500 hover:bg-gray-100'}`}
          >
            Original
          </button>
          <button
            onClick={() => setShowOverlay(true)}
            className={`px-3 py-1 rounded-lg ${showOverlay ? 'bg-primary-100 text-primary-700 font-medium' : 'text-gray-500 hover:bg-gray-100'}`}
          >
            Grad-CAM
          </button>
        </div>
      </div>
      <div className="rounded-xl overflow-hidden border border-gray-200">
        <img
          src={showOverlay ? gradcam : original}
          alt={showOverlay ? 'Grad-CAM overlay' : 'Imagem original'}
          className="w-full object-contain max-h-64"
        />
      </div>
    </div>
  )
}
