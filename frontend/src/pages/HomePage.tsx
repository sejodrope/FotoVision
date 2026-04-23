import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Loader2, Zap } from 'lucide-react'
import { ImageUploader } from '../components/diagnosis/ImageUploader'
import { DiagnosisCard } from '../components/diagnosis/DiagnosisCard'
import { runDiagnosis, getModelsStatus } from '../services/api'
import { useDiagnosisStore } from '../stores/diagnosisStore'
import { MODEL_LABELS } from '../types'
import type { ModelId } from '../types'

export function HomePage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const { selectedModel, generateGradcam, setSelectedModel, setGenerateGradcam, setLastResult, lastResult } =
    useDiagnosisStore()

  const { data: modelsStatus } = useQuery({
    queryKey: ['models'],
    queryFn: getModelsStatus,
  })

  const mutation = useMutation({
    mutationFn: () => runDiagnosis(file!, selectedModel, generateGradcam),
    onSuccess: (data) => setLastResult(data),
  })

  const handleFileSelected = (f: File, p: string) => {
    setFile(f)
    setPreview(p)
    mutation.reset()
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Diagnóstico Fitossanitário</h1>
        <p className="text-gray-500 mt-1">
          Envie uma foto da folha e o sistema detecta automaticamente anomalias.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-6">
          <ImageUploader onFileSelected={handleFileSelected} disabled={mutation.isPending} />

          <div className="bg-white rounded-2xl border border-gray-200 p-5 space-y-4">
            <h3 className="font-semibold text-gray-800">Configuração</h3>

            <div>
              <label className="block text-sm text-gray-600 mb-2">Modelo de IA</label>
              <div className="grid grid-cols-2 gap-2">
                {(Object.keys(MODEL_LABELS) as ModelId[]).map((id) => (
                  <button
                    key={id}
                    onClick={() => setSelectedModel(id)}
                    className={`py-2 px-3 rounded-xl text-sm font-medium border transition-all ${
                      selectedModel === id
                        ? 'border-primary-500 bg-primary-50 text-primary-700'
                        : 'border-gray-200 text-gray-600 hover:border-gray-300'
                    }`}
                  >
                    {MODEL_LABELS[id]}
                    {modelsStatus?.models.find((m) => m.id === id)?.calibrated && (
                      <span className="ml-1 text-xs text-green-600">✓</span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={generateGradcam}
                onChange={(e) => setGenerateGradcam(e.target.checked)}
                className="w-4 h-4 accent-green-600"
              />
              <span className="text-sm text-gray-700">Gerar mapa Grad-CAM (explicabilidade)</span>
            </label>

            <button
              onClick={() => mutation.mutate()}
              disabled={!file || mutation.isPending}
              className="w-full py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2"
            >
              {mutation.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analisando...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Analisar folha
                </>
              )}
            </button>

            {mutation.isError && (
              <p className="text-sm text-red-600 text-center">
                Erro ao processar. Verifique se o backend está rodando.
              </p>
            )}
          </div>
        </div>

        <div>
          {lastResult && preview ? (
            <DiagnosisCard result={lastResult} originalPreview={preview} />
          ) : (
            <div className="h-full min-h-[400px] flex items-center justify-center bg-gray-50 rounded-2xl border-2 border-dashed border-gray-200">
              <div className="text-center text-gray-400">
                <p className="text-lg font-medium">O resultado aparece aqui</p>
                <p className="text-sm mt-1">Selecione uma imagem e clique em Analisar</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
