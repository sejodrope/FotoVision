import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  AlertTriangle,
  CheckCircle,
  Loader2,
  RefreshCw,
  RotateCcw,
  WifiOff,
  FileX,
  Zap,
} from 'lucide-react'
import { ImageUploader } from '../components/diagnosis/ImageUploader'
import { runPredict } from '../services/api'
import type { PredictResult } from '../types'

type ApiError = {
  response?: { status?: number; data?: { detail?: string } }
  message?: string
}

function getErrorInfo(err: ApiError): { icon: React.ReactNode; text: string } {
  const status = err.response?.status
  const detail = err.response?.data?.detail
  if (!err.response)
    return {
      icon: <WifiOff className="w-5 h-5 shrink-0 text-red-500" />,
      text: 'Servidor offline ou sem resposta. Verifique se o backend está em execução.',
    }
  if (status === 415)
    return {
      icon: <FileX className="w-5 h-5 shrink-0 text-red-500" />,
      text: 'Arquivo não reconhecido como imagem. Use JPG, PNG ou WEBP.',
    }
  if (status === 413)
    return {
      icon: <FileX className="w-5 h-5 shrink-0 text-red-500" />,
      text: 'Imagem muito grande. O limite é 10 MB.',
    }
  if (status === 503)
    return {
      icon: <AlertTriangle className="w-5 h-5 shrink-0 text-red-500" />,
      text: 'Modelo não disponível no servidor.',
    }
  return {
    icon: <AlertTriangle className="w-5 h-5 shrink-0 text-red-500" />,
    text: detail ?? 'Erro inesperado. Tente novamente.',
  }
}

// ─── sub-components ──────────────────────────────────────────────────────────

function PlaceholderCard() {
  return (
    <div className="h-full min-h-[440px] flex items-center justify-center bg-gray-50 rounded-2xl border-2 border-dashed border-gray-200">
      <div className="text-center px-8">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
          <Zap className="w-8 h-8 text-gray-300" />
        </div>
        <p className="text-lg font-medium text-gray-500">O resultado aparece aqui</p>
        <p className="text-sm text-gray-400 mt-1">Selecione uma imagem e clique em Analisar</p>
      </div>
    </div>
  )
}

function LoadingCard({ preview }: { preview: string | null }) {
  return (
    <div className="relative rounded-2xl border border-gray-200 shadow-sm overflow-hidden min-h-[440px] bg-white flex items-center justify-center">
      {preview && (
        <img
          src={preview}
          alt=""
          className="absolute inset-0 w-full h-full object-cover opacity-10"
        />
      )}
      <div className="relative flex flex-col items-center gap-5 z-10">
        <div className="relative flex items-center justify-center">
          <div className="absolute w-16 h-16 rounded-full bg-primary-100 animate-ping opacity-40" />
          <div className="w-16 h-16 rounded-full bg-primary-50 border-2 border-primary-200 flex items-center justify-center">
            <Loader2 className="w-8 h-8 text-primary-600 animate-spin" />
          </div>
        </div>
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-800">Analisando imagem…</p>
          <p className="text-sm text-gray-500 mt-1">MobileNetV2 — classificação binária</p>
        </div>
      </div>
    </div>
  )
}

function ErrorBanner({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  const { icon, text } = getErrorInfo(error)
  return (
    <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-xl">
      {icon}
      <p className="flex-1 text-sm text-red-700 min-w-0">{text}</p>
      <button
        onClick={onRetry}
        className="shrink-0 flex items-center gap-1.5 text-xs font-medium text-red-600 hover:text-red-800 bg-white border border-red-200 rounded-lg px-2.5 py-1.5 transition-colors"
      >
        <RefreshCw className="w-3.5 h-3.5" />
        Tentar de novo
      </button>
    </div>
  )
}

function ProbBar({
  label,
  prob,
  color,
  isTop,
}: {
  label: string
  prob: number
  color: string
  isTop: boolean
}) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1.5">
        <span className={`text-sm ${isTop ? 'font-semibold text-gray-800' : 'font-medium text-gray-500'}`}>
          {label}
        </span>
        <span className="text-sm font-bold tabular-nums" style={{ color }}>
          {(prob * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.max(prob * 100, 0.5)}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function BinaryResultCard({
  result,
  originalPreview,
  onReset,
}: {
  result: PredictResult
  originalPreview: string
  onReset: () => void
}) {
  const isHealthy = result.label === 'healthy'
  const accentColor = isHealthy ? '#16a34a' : '#dc2626'
  const headerBg = isHealthy ? '#f0fdf4' : '#fef2f2'
  const headerBorder = isHealthy ? '#bbf7d0' : '#fecaca'

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      {/* verdict header */}
      <div
        className="px-6 py-5 border-b flex items-center gap-4"
        style={{ backgroundColor: headerBg, borderColor: headerBorder }}
      >
        <div
          className="w-14 h-14 shrink-0 rounded-2xl flex items-center justify-center"
          style={{ backgroundColor: `${accentColor}20` }}
        >
          {isHealthy ? (
            <CheckCircle className="w-8 h-8" style={{ color: accentColor }} />
          ) : (
            <AlertTriangle className="w-8 h-8" style={{ color: accentColor }} />
          )}
        </div>
        <div>
          <p
            className="text-xs font-semibold uppercase tracking-widest"
            style={{ color: accentColor }}
          >
            {isHealthy ? 'Planta saudável' : 'Anomalia detectada'}
          </p>
          <h2 className="text-2xl font-bold text-gray-900 leading-tight mt-0.5">
            {isHealthy ? 'Saudável' : 'Anômala'}
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Confiança:{' '}
            <span className="font-semibold" style={{ color: accentColor }}>
              {(result.confidence * 100).toFixed(1)}%
            </span>
          </p>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* probability bars */}
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
            Probabilidades por classe
          </p>
          <ProbBar label="Saudável" prob={result.healthy_prob} color="#16a34a" isTop={isHealthy} />
          <ProbBar
            label="Anômala"
            prob={result.anomalous_prob}
            color="#dc2626"
            isTop={!isHealthy}
          />
        </div>

        {/* model badge */}
        <div className="flex items-center justify-between text-xs text-gray-400 bg-gray-50 rounded-xl px-4 py-2.5">
          <span>MobileNetV2 — classificação binária</span>
          <span className="font-medium text-gray-500">98,81% acc.</span>
        </div>

        {/* image preview */}
        <div className="rounded-xl overflow-hidden border border-gray-100 max-h-52">
          <img
            src={originalPreview}
            alt="Folha analisada"
            className="w-full object-cover"
          />
        </div>

        {/* reset action */}
        <button
          onClick={onReset}
          className="w-full py-2.5 border border-gray-200 rounded-xl text-sm font-medium text-gray-600 hover:bg-gray-50 hover:border-gray-300 transition-colors flex items-center justify-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Tentar outra imagem
        </button>
      </div>
    </div>
  )
}

// ─── page ─────────────────────────────────────────────────────────────────────

export function HomePage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [uploaderKey, setUploaderKey] = useState(0)

  const mutation = useMutation({
    mutationFn: () => runPredict(file!),
    onSuccess: (data) => setResult(data),
  })

  const handleFileSelected = (f: File, p: string) => {
    setFile(f)
    setPreview(p)
    setResult(null)
    mutation.reset()
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setUploaderKey((k) => k + 1)
    mutation.reset()
  }

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Diagnóstico Fitossanitário</h1>
        <p className="text-gray-500 mt-1">
          Envie uma foto da folha e o sistema detecta automaticamente anomalias.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* ── coluna esquerda: upload + botão + erro ── */}
        <div className="space-y-4">
          <ImageUploader
            key={uploaderKey}
            onFileSelected={handleFileSelected}
            disabled={mutation.isPending}
          />

          <button
            onClick={() => mutation.mutate()}
            disabled={!file || mutation.isPending}
            className="w-full py-3.5 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2 text-base"
          >
            {mutation.isPending ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analisando…
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Analisar folha
              </>
            )}
          </button>

          {mutation.isError && (
            <ErrorBanner
              error={mutation.error as ApiError}
              onRetry={() => mutation.mutate()}
            />
          )}
        </div>

        {/* ── coluna direita: resultado / loading / placeholder ── */}
        <div>
          {mutation.isPending ? (
            <LoadingCard preview={preview} />
          ) : result && preview ? (
            <BinaryResultCard result={result} originalPreview={preview} onReset={handleReset} />
          ) : (
            <PlaceholderCard />
          )}
        </div>
      </div>
    </div>
  )
}
