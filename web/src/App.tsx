import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiError,
  autoSettle,
  getHealth,
  getLivePerformance,
  getModelPerformance,
  getRecentAlerts,
  getRecentPredictions,
  scanMarkets,
} from './api/client'
import type {
  AlertRecord,
  HealthResponse,
  MarketScanGame,
  ModelPerformance,
  PerformanceMap,
  RecentPrediction,
  Sport,
} from './api/types'
import { Alerts } from './components/Alerts'
import { Board } from './components/Board'
import { Header } from './components/Header'
import { Ticket } from './components/Ticket'
import { formatSettle, sortGames, timeAgo, type SortKey } from './lib/format'

type View = 'board' | 'ticket' | 'alerts'

const REFRESH_SEC = 300
let booted = false

function errMessage(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 0) return `Scan failed — ${e.body}. Is the API running on :8010?`
    if (e.status === 401) return 'Unauthorized — set VITE_API_KEY in web/.env.local to match API_KEY.'
    return e.body
  }
  return e instanceof Error ? e.message : 'Request failed'
}

export default function App() {
  const [view, setView] = useState<View>('board')
  const [sport, setSport] = useState<Sport | 'all'>('all')
  const [sort, setSort] = useState<SortKey>('edge')

  const [games, setGames] = useState<MarketScanGame[]>([])
  const [openKey, setOpenKey] = useState<string | null>(null)
  const [scanning, setScanning] = useState(false)
  const [scanError, setScanError] = useState<string | null>(null)
  const [scannedAt, setScannedAt] = useState<string | null>(null)
  const [refreshIn, setRefreshIn] = useState(0)

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [live, setLive] = useState<PerformanceMap | null>(null)
  const [models, setModels] = useState<ModelPerformance[]>([])
  const [recent, setRecent] = useState<RecentPrediction[]>([])
  const [alerts, setAlerts] = useState<AlertRecord[]>([])
  const [settling, setSettling] = useState(false)
  const [settleNote, setSettleNote] = useState<string | null>(null)

  const gamesRef = useRef(games)
  gamesRef.current = games
  const scanningRef = useRef(false)

  const loadMeta = useCallback(async () => {
    const [h, p, m, r, a] = await Promise.allSettled([
      getHealth(),
      getLivePerformance(),
      getModelPerformance(),
      getRecentPredictions(100),
      getRecentAlerts(50),
    ])
    if (h.status === 'fulfilled') setHealth(h.value)
    if (p.status === 'fulfilled') setLive(p.value)
    if (m.status === 'fulfilled') setModels(m.value)
    if (r.status === 'fulfilled') setRecent(r.value)
    if (a.status === 'fulfilled') setAlerts(a.value)
  }, [])

  const runScan = useCallback(async (isAuto = false) => {
    if (scanningRef.current) return
    scanningRef.current = true
    setScanning(true)
    if (!isAuto) setScanError(null)
    try {
      const data = await scanMarkets()
      setGames(data.results ?? [])
      setScannedAt(data.scanned_at)
      setScanError(null)
      setRefreshIn(REFRESH_SEC)
      await loadMeta()
    } catch (e) {
      if (!isAuto || gamesRef.current.length === 0) {
        setScanError(errMessage(e))
      } else {
        setScanError(`Refresh failed — keeping last board. ${errMessage(e)}`)
      }
      setRefreshIn(REFRESH_SEC)
    } finally {
      scanningRef.current = false
      setScanning(false)
    }
  }, [loadMeta])

  useEffect(() => {
    if (booted) return
    booted = true
    void loadMeta()
    void runScan(false)
  }, [loadMeta, runScan])

  useEffect(() => {
    if (refreshIn <= 0) return
    const id = window.setTimeout(() => {
      if (refreshIn <= 1) {
        if (!scanningRef.current) void runScan(true)
        else setRefreshIn(REFRESH_SEC)
      } else {
        setRefreshIn(refreshIn - 1)
      }
    }, 1000)
    return () => window.clearTimeout(id)
  }, [refreshIn, runScan])

  const visible = useMemo(() => {
    let list = games
    if (sport !== 'all') list = list.filter((g) => g.sport === sport)
    return sortGames(list, sort)
  }, [games, sport, sort])

  async function onSettle() {
    setSettling(true)
    try {
      const res = await autoSettle()
      setSettleNote(formatSettle(res))
      const [r, p] = await Promise.all([getRecentPredictions(100), getLivePerformance()])
      setRecent(r)
      setLive(p)
    } catch (e) {
      setSettleNote(errMessage(e))
    } finally {
      setSettling(false)
    }
  }

  return (
    <div className="app">
      <Header
        view={view}
        onView={setView}
        alertCount={alerts.length}
        sport={sport}
        sort={sort}
        onSport={setSport}
        onSort={setSort}
        health={health}
        live={live}
        models={models}
        scanning={scanning}
        lastScan={scannedAt ? timeAgo(scannedAt) : null}
        refreshIn={refreshIn}
        onScan={() => void runScan(false)}
      />
      {scanError && <div className="banner">{scanError}</div>}
      <div className="stage">
        {view === 'board' && (
          <Board
            games={visible}
            openKey={openKey}
            onToggle={(key) => setOpenKey((k) => (k === key ? null : key))}
            scanning={scanning && games.length === 0}
          />
        )}
        {view === 'ticket' && (
          <Ticket rows={recent} settling={settling} settleNote={settleNote} onSettle={() => void onSettle()} />
        )}
        {view === 'alerts' && <Alerts rows={alerts} />}
      </div>
    </div>
  )
}
