import { useQuery } from '@tanstack/react-query'
import { getModelsStatus, getHistory } from '../services/api'
import { ModelComparison } from '../components/dashboard/ModelComparison'
import { MetricsCard } from '../components/dashboard/MetricsCard'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { CLASS_COLORS, CLASS_LABELS_SHORT } from '../types'
import type { ClassId } from '../types'
import { Loader2 } from 'lucide-react'

export function DashboardPage() {
  const { data: models, isLoading: loadingModels } = useQuery({
    queryKey: ['models'],
    queryFn: getModelsStatus,
  })

  const { data: history, isLoading: loadingHistory } = useQuery({
    queryKey: ['history', 200],
    queryFn: () => getHistory(200),
  })

  const classCounts = history?.reduce(
    (acc, d) => {
      acc[d.predicted_class] = (acc[d.predicted_class] || 0) + 1
      return acc
    },
    {} as Record<string, number>,
  )

  const chartData = Object.entries(classCounts ?? {}).map(([cls, count]) => ({
    name: CLASS_LABELS_SHORT[cls as ClassId] ?? cls,
    count,
    cls,
  }))

  const avgConfidence =
    history?.length
      ? (history.reduce((s, d) => s + d.confidence, 0) / history.length) * 100
      : null

  const calibratedCount = models?.models.filter((m) => m.calibrated).length ?? 0

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Painel de Pesquisa</h1>
        <p className="text-gray-500 mt-1">Métricas dos modelos e distribuição dos diagnósticos.</p>
      </div>

      {loadingModels || loadingHistory ? (
        <div className="flex justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricsCard
              title="Diagnósticos realizados"
              value={history?.length ?? 0}
              subtitle="total acumulado"
            />
            <MetricsCard
              title="Modelos calibrados"
              value={`${calibratedCount}/4`}
              subtitle="pesos do TCC carregados"
              color={calibratedCount === 4 ? '#22c55e' : '#f59e0b'}
            />
            <MetricsCard
              title="Confiança média"
              value={avgConfidence !== null ? `${avgConfidence.toFixed(1)}%` : '—'}
              subtitle="sobre todos os modelos"
              color="#3b82f6"
            />
            <MetricsCard
              title="Modo"
              value={models?.demo_mode ? 'Demo' : 'Produção'}
              subtitle={models?.demo_mode ? 'pesos ImageNet' : 'modelos treinados'}
              color={models?.demo_mode ? '#f59e0b' : '#22c55e'}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {models && <ModelComparison models={models.models} />}

            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
              <h3 className="font-semibold text-gray-900 mb-1">Distribuição por classe</h3>
              <p className="text-sm text-gray-500 mb-6">Diagnósticos agrupados por anomalia</p>
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={chartData} barCategoryGap="30%">
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                      {chartData.map((entry) => (
                        <Cell
                          key={entry.cls}
                          fill={CLASS_COLORS[entry.cls as ClassId] ?? '#94a3b8'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-60 flex items-center justify-center text-gray-400 text-sm">
                  Nenhum diagnóstico ainda. Vá para a tela de diagnóstico e analise uma folha.
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
