import type { AlertRecord } from '../api/types'
import { evPct, f1, pct, timeAgo } from '../lib/format'

export function Alerts({ rows }: { rows: AlertRecord[] }) {
  if (!rows.length) {
    return <div className="empty">No high-confidence alerts recorded.</div>
  }

  return (
    <div className="sheet">
      <table className="board alerts">
        <thead>
          <tr>
            <th>When</th>
            <th>Matchup</th>
            <th>Sport</th>
            <th className="num">Line</th>
            <th className="num">Side</th>
            <th className="num">Conf</th>
            <th className="num">Edge</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((a) => (
            <tr key={a.alert_id}>
              <td className="sport-tag">{timeAgo(a.created_at)}</td>
              <td>
                <span className="matchup">
                  {a.away_team} @ {a.home_team}
                </span>
              </td>
              <td className="sport-tag">{a.sport}</td>
              <td className="num line">{f1(a.total_line)}</td>
              <td className={`num side ${a.direction}`}>{a.direction.toUpperCase()}</td>
              <td className="num conf">{pct(a.confidence, 0)}</td>
              <td className="num edge">{a.edge != null ? evPct(a.edge) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
