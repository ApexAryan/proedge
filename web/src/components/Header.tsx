import type { HealthResponse, ModelPerformance, PerformanceMap, Sport } from '../api/types'
import { SPORTS, type SortKey } from '../lib/format'
import { Logo } from './Logo'
import { SortMenu } from './SortMenu'

type View = 'board' | 'ticket' | 'alerts'

type Props = {
  view: View
  onView: (v: View) => void
  alertCount: number
  sport: Sport | 'all'
  sort: SortKey
  onSport: (s: Sport | 'all') => void
  onSort: (s: SortKey) => void
  health: HealthResponse | null
  live: PerformanceMap | null
  models: ModelPerformance[]
  scanning: boolean
  lastScan: string | null
  refreshIn: number
  onScan: () => void
}

const SPORT_OPTS: Array<Sport | 'all'> = ['all', 'nba', 'mlb', 'nfl']

function sportLabel(sport: Sport, live: PerformanceMap | null, models: ModelPerformance[]): string {
  const row = live?.[sport]
  if (row && row.total_settled > 0) return `${row.wins}–${row.losses}`
  const trained = models
    .filter((m) => m.sport === sport)
    .sort((a, b) => b.trained_at.localeCompare(a.trained_at))[0]
  if (trained?.accuracy != null) return `${Math.round(trained.accuracy * 100)}% tr`
  return '—'
}

function pnlSummary(live: PerformanceMap | null): { text: string; cls: string } | null {
  const o = live?.overall
  if (!o) return null
  const parts: string[] = []
  if (o.logged) parts.push(`${o.logged} logged`)
  if (o.pending) parts.push(`${o.pending} pending`)
  if (o.hit_rate != null) parts.push(`${(o.hit_rate * 100).toFixed(1)}% hit`)
  if (o.total_settled) parts.push(`${o.wins}–${o.losses}`)
  if (o.roi != null) {
    const pct = (o.roi * 100).toFixed(1)
    parts.push(`${o.roi >= 0 ? '+' : ''}${pct}% ROI`)
  }
  if (!parts.length) return null
  return {
    text: parts.join('  ·  '),
    cls: (o.roi ?? 0) > 0 ? 'pos' : (o.roi ?? 0) < 0 ? 'neg' : '',
  }
}

export function Header({
  view,
  onView,
  alertCount,
  sport,
  sort,
  onSport,
  onSort,
  health,
  live,
  models,
  scanning,
  lastScan,
  refreshIn,
  onScan,
}: Props) {
  const status = health?.status ?? '…'
  const m = Math.floor(refreshIn / 60)
  const s = String(refreshIn % 60).padStart(2, '0')
  const pnl = pnlSummary(live)

  return (
    <header className="chrome">
      <div className="chrome-top">
        <Logo />
        {pnl && (
          <span className={`kpis ${pnl.cls}`} title="per unit at −110">
            {pnl.text}
          </span>
        )}
        <div className="chrome-top-right">
          <span className={`health ${status}`}>
            <i />
            {status}
          </span>
          {lastScan && <span>{lastScan}</span>}
          {refreshIn > 0 && <span>{m}:{s}</span>}
          <button className="scan-btn" onClick={onScan} disabled={scanning}>
            {scanning ? 'Scanning' : 'Scan'}
          </button>
        </div>
      </div>
      <div className="chrome-bar">
        <nav className="views">
          <button className={view === 'board' ? 'on' : ''} onClick={() => onView('board')}>
            Board
          </button>
          <button className={view === 'ticket' ? 'on' : ''} onClick={() => onView('ticket')}>
            Ticket
          </button>
          <button className={view === 'alerts' ? 'on' : ''} onClick={() => onView('alerts')}>
            Alerts{alertCount ? ` ${alertCount}` : ''}
          </button>
        </nav>
        {view === 'board' && (
          <div className="filters">
            {SPORT_OPTS.map((sp) => (
              <button key={sp} className={sport === sp ? 'on' : ''} onClick={() => onSport(sp)}>
                {sp === 'all' ? 'All' : sp.toUpperCase()}
              </button>
            ))}
          </div>
        )}
        <div className="chrome-bar-right">
          <div className="sport-stats">
            {SPORTS.map((sp) => (
              <span key={sp}>
                {sp.toUpperCase()} <b>{sportLabel(sp, live, models)}</b>
              </span>
            ))}
          </div>
          {view === 'board' && <SortMenu value={sort} onChange={onSort} />}
        </div>
      </div>
    </header>
  )
}
