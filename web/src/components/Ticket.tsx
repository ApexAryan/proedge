import type { RecentPrediction } from '../api/types'
import { f1, isLocalDate, money, paperPnl, pct, timeAgo } from '../lib/format'

type Props = {
  rows: RecentPrediction[]
  settling: boolean
  settleNote: string | null
  onSettle: () => void
}

export function Ticket({ rows, settling, settleNote, onSettle }: Props) {
  const today = rows.filter((p) => isLocalDate(p.game_date) || isLocalDate(p.predicted_at))
  const shown = today.length ? today : rows.slice(0, 20)
  const pending = shown.filter((p) => p.is_correct == null).length

  return (
    <>
      <div className="sec-bar">
        <span>{shown.length} {today.length ? 'today' : 'recent'}</span>
        <span>{pending} pending</span>
        {settleNote && <span className="settle-note">{settleNote}</span>}
        <button className="settle-btn" onClick={onSettle} disabled={settling}>
          {settling ? 'Settling…' : 'Settle'}
        </button>
      </div>
      {!shown.length ? (
        <div className="empty">No predictions yet. Scan to write today’s ticket.</div>
      ) : (
        <div className="sheet">
          <table className="board ticket">
            <thead>
              <tr>
                <th>Matchup</th>
                <th>Sport</th>
                <th className="num">Line</th>
                <th className="num">Call</th>
                <th className="num">Actual</th>
                <th className="num">CLV</th>
                <th className="num">Result</th>
                <th className="num">P&amp;L</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((p) => {
                const pnl = paperPnl(p.is_correct)
                const result =
                  p.is_correct == null ? 'pending' : p.is_correct ? 'win' : 'loss'
                return (
                  <tr key={p.prediction_id}>
                    <td>
                      <span className="matchup">
                        {p.away_team} @ {p.home_team}
                      </span>
                      <div className="sport-tag">{timeAgo(p.predicted_at)}</div>
                    </td>
                    <td className="sport-tag">{p.sport}</td>
                    <td className="num line">{f1(p.total_line)}</td>
                    <td className={`num side ${p.predicted_direction}`}>
                      {p.predicted_direction.toUpperCase()}{' '}
                      {pct(p.predicted_direction === 'over' ? p.prob_over : p.prob_under, 0)}
                    </td>
                    <td className="num conf">
                      {p.actual_total != null ? f1(p.actual_total) : '—'}
                      {p.closing_line != null ? ` / ${f1(p.closing_line)}` : ''}
                    </td>
                    <td className="num conf">{p.clv != null ? f1(p.clv) : '—'}</td>
                    <td className="num">
                      <span className={`result ${result}`}>
                        {result === 'pending' ? 'open' : result}
                      </span>
                    </td>
                    <td className={`num ${pnl != null && pnl > 0 ? 'pos-ev' : pnl != null && pnl < 0 ? 'result loss' : 'muted'}`}>
                      {money(pnl)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}
