interface Props {
  title: string
  value: string | number
  subtitle?: string
  color?: string
}

export function MetricsCard({ title, value, subtitle, color = '#22c55e' }: Props) {
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
      <p className="text-sm text-gray-500">{title}</p>
      <p className="text-3xl font-bold mt-1" style={{ color }}>
        {value}
      </p>
      {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
    </div>
  )
}
