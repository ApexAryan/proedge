import { useEffect, useState } from 'react'
import { getLineMatrix } from '../api/client'
import type { LineMatrix, MarketScanGame } from '../api/types'
import { cents, evPct, f1, pct } from '../lib/format'

export function GameDetail({ game }: { game: MarketScanGame }) {
  const [matrix, setMatrix] = useState<LineMatrix | null>(null)

  useEffect(() => {
    let cancelled = false
    getLineMatrix(game.sport, game.home_team, game.away_team)
      .then((m) => {
        if (!cancelled) setMatrix(m)
      })
      .catch(() => {
        if (!cancelled) setMatrix(null)
      })
    return () => {
      cancelled = true
    }
  }, [game.sport, game.home_team, game.away_team])

  const hasEdge = game.best_bet != null && game.best_edge != null
  const implied = game.kalshi_implied_line
  const nearestIdx = game.thresholds.reduce((best, t, i, arr) => {
    if (!arr.length) return -1
    return Math.abs(t.threshold - implied) < Math.abs(arr[best]?.threshold - implied) ? i : best
  }, 0)

  return (
    <div className={`detail ${hasEdge ? 'has-edge' : ''}`}>
      <div className="call">
        <div>
          <div className={`call-main side ${game.best_bet ?? ''}`}>
            {hasEdge
              ? `${game.best_bet?.toUpperCase()} ${game.best_threshold}`
              : 'No edge'}
          </div>
          <div className="call-sub">
            {hasEdge
              ? `Kalshi · ${evPct(game.best_edge)} EV`
              : 'Market priced fairly'}
            {game.line_movement != null && Math.abs(game.line_movement) >= 0.05
              ? ` · line ${game.line_movement > 0 ? '+' : ''}${f1(game.line_movement)} since first seen`
              : ''}
          </div>
        </div>
        <div className="stat">
          Over <b className="side over">{pct(game.model_prob_over)}</b>
        </div>
        <div className="stat">
          Under <b className="side under">{pct(game.model_prob_under)}</b>
        </div>
        <div className="stat">
          Conf <b>{Math.round(game.confidence * 100)}%</b>
        </div>
        <div className="stat">
          P(over) CI{' '}
          <b>
            {game.ci_lower != null && game.ci_upper != null
              ? `${pct(game.ci_lower)}–${pct(game.ci_upper)}`
              : '—'}
          </b>
        </div>
      </div>

      <div className="books">
        <div className="book">
          Kalshi
          <b>{f1(matrix?.kalshi_implied_line ?? game.kalshi_implied_line)}</b>
        </div>
        <div className="book">
          PrizePicks
          <b>{f1(matrix?.prizepicks_line)}</b>
        </div>
        <div className="book">
          Underdog
          <b>
            {matrix?.underdog_line != null
              ? `${f1(matrix.underdog_line)}${matrix.underdog_over_american ? `  O ${matrix.underdog_over_american}` : ''}${matrix.underdog_under_american ? `  U ${matrix.underdog_under_american}` : ''}`
              : '—'}
          </b>
        </div>
      </div>

      <div className="meta-row">
        <span>
          Injuries {game.home_team} {game.home_key_players_out} out · {game.away_team}{' '}
          {game.away_key_players_out} out
        </span>
        <span>{game.live_stats ? 'Live team stats' : 'Median features'}</span>
      </div>

      {game.thresholds.length ? (
        <div className="matrix-scroll">
        <table className="matrix">
          <thead>
            <tr>
              <th>Line</th>
              <th>Over ask</th>
              <th>Under ask</th>
              <th>Market</th>
              <th>Model</th>
              <th>EV over</th>
              <th>EV under</th>
            </tr>
          </thead>
          <tbody>
            {game.thresholds.map((t, i) => {
              const illiquid = t.market_prob <= 0.12 || t.market_prob >= 0.88
              const far = Math.abs(t.threshold - implied) > 3
              const cls = [
                t.best_bet ? 'best' : '',
                i === nearestIdx ? 'implied' : '',
                far ? 'dim' : '',
              ]
                .filter(Boolean)
                .join(' ')
              return (
                <tr key={`${t.threshold}-${i}`} className={cls}>
                  <td>
                    {t.threshold}
                    {i === nearestIdx ? ' · 50%' : ''}
                    {illiquid ? ' · illiq' : ''}
                  </td>
                  <td>{cents(t.yes_ask)}</td>
                  <td>{cents(t.no_ask)}</td>
                  <td>{pct(t.market_prob, 0)}</td>
                  <td>{pct(t.model_prob, 0)}</td>
                  <td className={t.ev_yes != null && t.ev_yes > 0 ? 'pos-ev' : 'muted'}>
                    {evPct(t.ev_yes)}
                  </td>
                  <td className={t.ev_no != null && t.ev_no > 0 ? 'pos-ev' : 'muted'}>
                    {evPct(t.ev_no)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        </div>
      ) : (
        <div className="muted">No Kalshi thresholds for this game.</div>
      )}
    </div>
  )
}
