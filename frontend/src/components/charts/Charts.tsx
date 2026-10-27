import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const AXIS = { fontSize: 11, fill: 'var(--muted)' }
const GRID = 'var(--line)'

export interface Datum {
  name: string
  value: number
  color?: string
}

/** Horizontal bars; height grows with the number of rows so labels never overlap. */
export function HBarChart({
  data,
  color = 'var(--accent)',
  format = (v: number) => String(v),
  labelWidth = 130,
  onSelect,
  ariaLabel,
}: {
  data: Datum[]
  color?: string
  format?: (v: number) => string
  labelWidth?: number
  onSelect?: (name: string) => void
  ariaLabel: string
}) {
  const height = Math.max(120, data.length * 26 + 24)
  return (
    <div role="img" aria-label={ariaLabel} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 36, bottom: 4, left: 4 }}>
          <CartesianGrid horizontal={false} stroke={GRID} />
          <XAxis type="number" tick={AXIS} tickFormatter={format} stroke={GRID} allowDecimals />
          <YAxis type="category" dataKey="name" width={labelWidth} tick={AXIS} stroke={GRID} interval={0} />
          <Tooltip cursor={{ fill: 'var(--surface-2)' }} formatter={(v) => format(Number(v))} labelStyle={{ color: 'var(--fg)' }} />
          <Bar
            dataKey="value"
            radius={[0, 3, 3, 0]}
            maxBarSize={16}
            isAnimationActive={false}
            onClick={onSelect ? (d) => onSelect(String((d as { name?: string }).name)) : undefined}
            cursor={onSelect ? 'pointer' : undefined}
            label={{ position: 'right', fontSize: 11, fill: 'var(--muted)', formatter: (v: unknown) => format(Number(v)) }}
          >
            {data.map((d) => (
              <Cell key={d.name} fill={d.color ?? color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export interface SeriesSpec {
  key: string
  label: string
  color: string
}

export function GroupedBarChart({
  data,
  series,
  format = (v: number) => v.toFixed(3),
  domain,
  ariaLabel,
  height = 260,
}: {
  data: Record<string, string | number>[]
  series: SeriesSpec[]
  format?: (v: number) => string
  domain?: [number, number]
  ariaLabel: string
  height?: number
}) {
  return (
    <div role="img" aria-label={ariaLabel} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -8 }}>
          <CartesianGrid vertical={false} stroke={GRID} />
          <XAxis dataKey="name" tick={AXIS} stroke={GRID} interval={0} />
          <YAxis tick={AXIS} stroke={GRID} domain={domain} ticks={domain ? [0, 0.25, 0.5, 0.75, 1].map((t) => domain[0] + t * (domain[1] - domain[0])) : undefined} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip cursor={{ fill: 'var(--surface-2)' }} formatter={(v) => format(Number(v))} labelStyle={{ color: 'var(--fg)' }} />
          <Legend wrapperStyle={{ fontSize: 12, color: 'var(--muted)' }} iconType="square" iconSize={10} itemSorter={null} />
          {series.map((s) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[3, 3, 0, 0]} maxBarSize={28} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
