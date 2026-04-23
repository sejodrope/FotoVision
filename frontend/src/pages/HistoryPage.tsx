import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getHistory, deleteDiagnosis } from '../services/api'
import { CLASS_COLORS, MODEL_LABELS } from '../types'
import type { ClassId, ModelId } from '../types'
import { Trash2, Loader2, FlaskConical } from 'lucide-react'
import clsx from 'clsx'

const CLASS_LABELS: Record<ClassId, string> = {
  saudavel: 'Saudável',
  mildio: 'Míldio',
  oidio: 'Oídio',
  clorose_nitrogenio: 'Clorose por Nitrogênio',
  danos_pragas: 'Danos por Pragas',
  estresse_hidrico: 'Estresse Hídrico',
}

export function HistoryPage() {
  const queryClient = useQueryClient()
  const [deletingId, setDeletingId] = useState<number | null>(null)

  const { data: history, isLoading } = useQuery({
    queryKey: ['history', 50],
    queryFn: () => getHistory(50),
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDiagnosis,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['history'] }),
    onSettled: () => setDeletingId(null),
  })

  const handleDelete = (id: number) => {
    setDeletingId(id)
    deleteMutation.mutate(id)
  }

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Histórico de Diagnósticos</h1>
        <p className="text-gray-500 mt-1">Todos os diagnósticos realizados nesta sessão.</p>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
        </div>
      ) : !history?.length ? (
        <div className="text-center py-20 text-gray-400">
          <p className="text-lg font-medium">Nenhum diagnóstico ainda</p>
          <p className="text-sm mt-1">Vá para a página inicial e analise uma folha.</p>
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="text-left px-4 py-3 font-medium text-gray-600">Data</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">Arquivo</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">Modelo</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600">Diagnóstico</th>
                <th className="text-right px-4 py-3 font-medium text-gray-600">Confiança</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {history.map((item) => (
                <tr key={item.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3 text-gray-500 whitespace-nowrap">
                    {new Date(item.created_at).toLocaleString('pt-BR', {
                      day: '2-digit',
                      month: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </td>
                  <td className="px-4 py-3 text-gray-700 max-w-[160px] truncate">
                    {item.original_filename ?? '—'}
                  </td>
                  <td className="px-4 py-3 text-gray-700">
                    {MODEL_LABELS[item.model_name as ModelId] ?? item.model_name}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{
                          backgroundColor: CLASS_COLORS[item.predicted_class as ClassId] ?? '#94a3b8',
                        }}
                      />
                      <span className="text-gray-800">
                        {CLASS_LABELS[item.predicted_class as ClassId] ?? item.predicted_label}
                      </span>
                      {item.demo_mode && (
                        <FlaskConical className="w-3.5 h-3.5 text-amber-400" title="Modo demo" />
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right font-medium text-gray-800">
                    {(item.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={() => handleDelete(item.id)}
                      disabled={deletingId === item.id}
                      className={clsx(
                        'p-1.5 rounded-lg hover:bg-red-50 text-gray-400 hover:text-red-500 transition-colors',
                        deletingId === item.id && 'opacity-50',
                      )}
                    >
                      {deletingId === item.id ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
