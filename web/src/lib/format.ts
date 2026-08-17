import type { MarketScanGame, SettleResult, Side, Sport } from '../api/types'

export function pct(v: number | null | undefined, digits = 1): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

export function f1(v: number | null | undefined): string {
  if (v == null) return '—'
  return Number(v).toFixed(1)
}

export function evPct(v: number | null | undefined): string {
  if (v == null) return '—'
  const n = v * 100
  return `${n >= 0 ? '+' : ''}${n.toFixed(1)}%`
}

export function money(v: number | null | undefined): string {
  if (v == null) return '—'
  const sign = v >= 0 ? '+' : '−'
  return `${sign}$${Math.abs(v).toFixed(2)}`
}

export function cents(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${Math.round(v * 100)}¢`
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return ''
  const s = (Date.now() - new Date(iso).getTime()) / 1000
  if (s < 60) return `${Math.round(s)}s ago`
  if (s < 3600) return `${Math.round(s / 60)}m ago`
  if (s < 86400) return `${Math.round(s / 3600)}h ago`
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export function isLocalDate(iso: string | null | undefined, day: Date = new Date()): boolean {
  if (!iso) return false
  const d = new Date(iso)
  return (
    d.getFullYear() === day.getFullYear() &&
    d.getMonth() === day.getMonth() &&
    d.getDate() === day.getDate()
  )
}

export function gameKey(g: { sport: string; home_team: string; away_team: string }): string {
  return `${g.sport}:${g.home_team}:${g.away_team}`
}

export function sideProb(g: MarketScanGame, side?: Side): number {
  const dir = side ?? g.predicted_direction
  return dir === 'over' ? g.model_prob_over : g.model_prob_under
}

export function paperPnl(isCorrect: boolean | null): number | null {
  if (isCorrect == null) return null
  return isCorrect ? 90.91 : -100
}

export const SPORTS: Sport[] = ['nba', 'mlb', 'nfl']

export type SortKey = 'edge' | 'confidence' | 'move'

export function sortGames(games: MarketScanGame[], sort: SortKey): MarketScanGame[] {
  const copy = [...games]
  copy.sort((a, b) => {
    if (sort === 'edge') return (b.best_edge ?? -99) - (a.best_edge ?? -99)
    if (sort === 'move') return Math.abs(b.line_movement ?? 0) - Math.abs(a.line_movement ?? 0)
    return b.confidence - a.confidence
  })
  return copy
}

export function formatSettle(res: SettleResult): string {
  if (res.error) return res.error
  const settled = res.settled ?? 0
  const already = res.already_settled ?? 0
  const miss = res.no_score_match ?? 0
  return `settled ${settled} · already ${already} · no score ${miss}`
}
