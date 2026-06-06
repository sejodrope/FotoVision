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
  Leaf,
  Database,
  Cpu,
} from 'lucide-react'
import { ImageUploader } from '../components/diagnosis/ImageUploader'
import { runPredict } from '../services/api'
import type { PredictResult } from '../types'

// ─── modelo activo ────────────────────────────────────────────────────────────
const MODEL_NAME = 'EfficientNet-B0'
const MODEL_ACC  = '99,01%'

// ─── tipos ────────────────────────────────────────────────────────────────────
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

// ─── stat item ────────────────────────────────────────────────────────────────
function StatItem({
  icon: Icon,
  value,
  label,
}: {
  icon: React.ElementType
  value: string
  label: string
}) {
  return (
    <div className="flex items-center gap-3 bg-white rounded-xl border border-gray-100 shadow-sm px-4 py-3">
      <div className="w-8 h-8 rounded-lg bg-primary-50 flex items-center justify-center shrink-0">
        <Icon className="w-4 h-4 text-primary-600" />
      </div>
      <div className="min-w-0">
        <p className="text-sm font-bold text-gray-900 leading-tight">{value}</p>
        <p className="text-xs text-gray-400 mt-0.5 truncate">{label}</p>
      </div>
    </div>
  )
}

// ─── gauge circular SVG ───────────────────────────────────────────────────────
function ConfidenceGauge({ value, color }: { value: number; color: string }) {
  const size   = 96
  const stroke = 8
  const r      = (size - stroke) / 2
  const circ   = 2 * Math.PI * r
  const offset = circ * (1 - value)

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke="#e5e7eb" strokeWidth={stroke}
        />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none" stroke={color} strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.7s ease-out' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className="text-lg font-bold tabular-nums leading-none"
          style={{ color }}
        >
          {(value * 100).toFixed(0)}%
        </span>
        <span className="text-[10px] text-gray-400 mt-0.5">confiança</span>
      </div>
    </div>
  )
}

// ─── placeholder ──────────────────────────────────────────────────────────────
function PlaceholderCard() {
  return (
    <div className="h-full min-h-[440px] flex items-center justify-center bg-gradient-to-br from-gray-50 to-white rounded-2xl border-2 border-dashed border-gray-200">
      <div className="text-center px-8">
        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-primary-50 flex items-center justify-center">
          <Leaf className="w-8 h-8 text-primary-300" />
        </div>
        <p className="text-base font-semibold text-gray-500">Resultado da análise</p>
        <p className="text-sm text-gray-400 mt-1">
          Envie uma imagem para iniciar o diagnóstico
        </p>
      </div>
    </div>
  )
}

// ─── loading ──────────────────────────────────────────────────────────────────
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
          <p className="text-sm text-gray-500 mt-1">{MODEL_NAME} · classificação binária</p>
        </div>
      </div>
    </div>
  )
}

// ─── erro ─────────────────────────────────────────────────────────────────────
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

// ─── barra de probabilidade ───────────────────────────────────────────────────
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
        <span
          className={`text-sm ${
            isTop ? 'font-semibold text-gray-800' : 'font-medium text-gray-500'
          }`}
        >
          {label}
        </span>
        <span className="text-sm font-bold tabular-nums" style={{ color }}>
          {(prob * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${Math.max(prob * 100, 0.5)}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

// ─── card de resultado ────────────────────────────────────────────────────────
function BinaryResultCard({
  result,
  originalPreview,
  onReset,
}: {
  result: PredictResult
  originalPreview: string
  onReset: () => void
}) {
  const isHealthy   = result.label === 'healthy'
  const accentColor = isHealthy ? '#267350' : '#dc2626'
  const headerBg    = isHealthy ? '#f0f7f3' : '#fef2f2'
  const headerBorder = isHealthy ? '#b3dcc6' : '#fecaca'
  const healthyColor = '#267350'
  const anomColor    = '#dc2626'

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      {/* cabeçalho com gauge */}
      <div
        className="px-6 py-5 border-b flex items-center gap-5"
        style={{ backgroundColor: headerBg, borderColor: headerBorder }}
      >
        <ConfidenceGauge value={result.confidence} color={accentColor} />
        <div className="flex-1 min-w-0">
          <p
            className="text-xs font-semibold uppercase tracking-widest"
            style={{ color: accentColor }}
          >
            {isHealthy ? 'Planta saudável' : 'Anomalia detectada'}
          </p>
          <h2 className="text-2xl font-bold text-gray-900 leading-tight mt-0.5">
            {isHealthy ? 'Saudável' : 'Anômala'}
          </h2>
          <div className="flex items-center gap-1.5 mt-1.5">
            {isHealthy ? (
              <CheckCircle className="w-3.5 h-3.5 shrink-0" style={{ color: accentColor }} />
            ) : (
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" style={{ color: accentColor }} />
            )}
            <p className="text-xs text-gray-500">
              Modelo:{' '}
              <span className="font-medium text-gray-700">{MODEL_NAME}</span>
            </p>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-4">
        {/* barras de probabilidade */}
        <div className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
            Distribuição de probabilidades
          </p>
          <ProbBar
            label="Saudável"
            prob={result.healthy_prob}
            color={healthyColor}
            isTop={isHealthy}
          />
          <ProbBar
            label="Anômala"
            prob={result.anomalous_prob}
            color={anomColor}
            isTop={!isHealthy}
          />
        </div>

        {/* prévia da imagem */}
        <div className="rounded-xl overflow-hidden border border-gray-100 max-h-48">
          <img
            src={originalPreview}
            alt="Folha analisada"
            className="w-full object-cover"
          />
        </div>

        {/* badge do modelo */}
        <div className="flex items-center justify-between text-xs bg-gray-50 rounded-xl px-4 py-2.5">
          <span className="text-gray-400">{MODEL_NAME} · classificação binária</span>
          <span className="font-semibold text-gray-600">{MODEL_ACC} acc.</span>
        </div>

        {/* nova análise */}
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

// ─── página ───────────────────────────────────────────────────────────────────
export function HomePage() {
  const [file, setFile]           = useState<File | null>(null)
  const [preview, setPreview]     = useState<string | null>(null)
  const [result, setResult]       = useState<PredictResult | null>(null)
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
      {/* hero */}
      <div className="mb-8">
        <div className="flex items-start gap-4">
          <div className="w-11 h-11 rounded-2xl bg-primary-600 flex items-center justify-center shrink-0 mt-0.5">
            <Leaf className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Diagnóstico Fitossanitário
            </h1>
            <p className="text-sm text-gray-500 mt-0.5">
              Detecção automática de anomalias em folhosas · Transfer Learning ·
              Classificação binária
            </p>
          </div>
        </div>

        {/* strip de métricas */}
        <div className="mt-5 grid grid-cols-3 gap-3">
          <StatItem icon={Zap}      value={MODEL_ACC}   label="Acurácia no test set" />
          <StatItem icon={Database} value="210.832"     label="Imagens de treino"    />
          <StatItem icon={Cpu}      value={MODEL_NAME}  label="Modelo activo"        />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
        {/* coluna esquerda: upload + botão + erro */}
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

        {/* coluna direita: resultado / loading / placeholder */}
        <div>
          {mutation.isPending ? (
            <LoadingCard preview={preview} />
          ) : result && preview ? (
            <BinaryResultCard
              result={result}
              originalPreview={preview}
              onReset={handleReset}
            />
          ) : (
            <PlaceholderCard />
          )}
        </div>
      </div>
    </div>
  )
}
