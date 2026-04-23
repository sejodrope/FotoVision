import type { ClassId } from '../../types'
import { CLASS_COLORS } from '../../types'
import clsx from 'clsx'

const CLASS_LABELS: Record<ClassId, string> = {
  saudavel: 'Saudável',
  mildio: 'Míldio',
  oidio: 'Oídio',
  clorose_nitrogenio: 'Clorose por Nitrogênio',
  danos_pragas: 'Danos por Pragas',
  estresse_hidrico: 'Estresse Hídrico',
}

interface Props {
  classId: ClassId
  score: number
  isTop?: boolean
}

export function ConfidenceBar({ classId, score, isTop }: Props) {
  const pct = (score * 100).toFixed(1)
  const color = CLASS_COLORS[classId]

  return (
    <div className={clsx('flex items-center gap-3', isTop && 'font-semibold')}>
      <span className="w-44 text-sm text-gray-700 truncate">{CLASS_LABELS[classId]}</span>
      <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.max(score * 100, 0.5)}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-14 text-sm text-right text-gray-600">{pct}%</span>
    </div>
  )
}
