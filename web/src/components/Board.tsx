import { Fragment } from 'react'
import type { MarketScanGame } from '../api/types'
import { evPct, f1, gameKey, pct, sideProb } from '../lib/format'
import { GameDetail } from './GameDetail'

type Props = {
  games: MarketScanGame[]
  openKey: string | null
  onToggle: (key: string) => void
  scanning: boolean
}

export function Board({ games, openKey, onToggle, scanning }: Props) {
  if (!games.length && scanning) {
    return <div className="loading">Querying Kalshi and scoring against the model…</div>
  }
  if (!games.length) {
    return <div className="empty">No open markets. Try again closer to game time.</div>
  }

  return (
    <div className="sheet">
      <table className="board">
        <colgroup>
          <col className="m" />
          <col className="s" />
          <col className="n" />
          <col className="d" />
          <col className="c" />
          <col className="o" />
          <col className="e" />
        </colgroup>
        <thead>
          <tr>
            <th>Matchup</th>
            <th>Sport</th>
            <th className="num">Line</th>
            <th className="num">Model</th>
            <th className="num">Conf</th>
            <th className="num">Out</th>
            <th className="num">Edge</th>
          </tr>
        </thead>
        <tbody>
          {games.map((g) => {
            const key = gameKey(g)
            const open = openKey === key
            const mv = g.line_movement
            const hasEdge = g.best_bet != null && g.best_edge != null
            return (
              <Fragment key={key}>
                <tr
                  className={`game ${open ? 'open' : ''}`}
                  onClick={() => onToggle(key)}
                >
                  <td>
                    <span className="matchup">
                      {g.away_team} @ {g.home_team}
                    </span>
                  </td>
                  <td className="sport-tag">{g.sport}</td>
                  <td className="num line">
                    {f1(g.kalshi_implied_line)}
                    {mv != null && Math.abs(mv) >= 0.05 && (
                      <span className={`move ${mv > 0 ? 'up' : 'down'}`}>
                        {mv > 0 ? '+' : ''}
                        {f1(mv)}
                      </span>
                    )}
                  </td>
                  <td className={`num side ${g.predicted_direction}`}>
                    {g.predicted_direction.toUpperCase()} {pct(sideProb(g), 0)}
                  </td>
                  <td className="num conf">{Math.round(g.confidence * 100)}%</td>
                  <td className="num sport-tag">
                    {g.home_team} {g.home_key_players_out} · {g.away_team} {g.away_key_players_out}
                  </td>
                  <td className="num">
                    {hasEdge ? (
                      <span className="edge">
                        {g.best_bet?.toUpperCase()} {g.best_threshold} {evPct(g.best_edge)}
                      </span>
                    ) : (
                      <span className="fair">fair</span>
                    )}
                  </td>
                </tr>
                {open && (
                  <tr className="detail-row">
                    <td
                      className="detail-cell"
                      colSpan={7}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <GameDetail game={g} />
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
