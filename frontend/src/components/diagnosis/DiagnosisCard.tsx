import { AlertTriangle, CheckCircle, FlaskConical } from 'lucide-react'
import type { DiagnosisResult, ClassId } from '../../types'
import { CLASS_COLORS, MODEL_LABELS } from '../../types'
import { ConfidenceBar } from './ConfidenceBar'
import { GradCamViewer } from './GradCamViewer'

interface Props {
  result: DiagnosisResult
  originalPreview: string
}

export function DiagnosisCard({ result, originalPreview }: Props) {
  const isHealthy = result.predicted_class === 'saudavel'
  const color = CLASS_COLORS[result.predicted_class as ClassId]
  const sortedScores = Object.entries(result.scores).sort(([, a], [, b]) => b - a) as [ClassId, number][]

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      {result.demo_mode && (
        <div className="bg-amber-50 border-b border-amber-200 px-4 py-2 flex items-center gap-2 text-sm text-amber-700">
          <FlaskConical className="w-4 h-4 shrink-0" />
          <span>
            <strong>Modo demonstração</strong> — modelo usando pesos ImageNet. Resultados não
            representam diagnóstico real. Aguardando treinamento com datasets fitossanitários.
          </span>
        </div>
      )}

      <div className="p-6 space-y-6">
        <div className="flex items-start gap-4">
          <div
            className="flex-shrink-0 w-14 h-14 rounded-2xl flex items-center justify-center"
            style={{ backgroundColor: `${color}20` }}
          >
            {isHealthy ? (
              <CheckCircle className="w-7 h-7" style={{ color }} />
            ) : (
              <AlertTriangle className="w-7 h-7" style={{ color }} />
            )}
          </div>
          <div>
            <p className="text-sm text-gray-500">Diagnóstico — {MODEL_LABELS[result.model_name]}</p>
            <h2 className="text-2xl font-bold text-gray-900">{result.predicted_label}</h2>
            <p className="text-sm font-medium mt-0.5" style={{ color }}>
              Confiança: {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="space-y-2">
          <p className="text-sm font-medium text-gray-700 mb-3">Probabilidades por classe</p>
          {sortedScores.map(([cls, score]) => (
            <ConfidenceBar
              key={cls}
              classId={cls}
              score={score}
              isTop={cls === result.predicted_class}
            />
          ))}
        </div>

        {result.gradcam_image && (
          <GradCamViewer
            original={originalPreview}
            gradcam={result.gradcam_image}
            modelLabel={MODEL_LABELS[result.model_name]}
          />
        )}
      </div>
    </div>
  )
}
