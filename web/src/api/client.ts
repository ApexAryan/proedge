import type {
  AlertRecord,
  HealthResponse,
  LineMatrix,
  MarketScanResponse,
  ModelPerformance,
  PerformanceMap,
  RecentPrediction,
  SettleResult,
  Sport,
} from './types'

export class ApiError extends Error {
  status: number
  body: string

  constructor(status: number, body: string) {
    super(body || `HTTP ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

async function request<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers = new Headers(opts.headers)
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const key = import.meta.env.VITE_API_KEY
  if (key) headers.set('X-API-Key', key)

  let res: Response
  try {
    res = await fetch(path, { ...opts, headers, credentials: 'include' })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Network error'
    throw new ApiError(0, msg)
  }

  if (!res.ok) {
    const text = await res.text()
    let detail = text
    try {
      const parsed = JSON.parse(text) as { detail?: string }
      if (parsed.detail) detail = parsed.detail
    } catch {
      /* raw body */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export function scanMarkets(sports: Sport[] = ['nba', 'mlb', 'nfl']): Promise<MarketScanResponse> {
  return request('/predictions/scan', {
    method: 'POST',
    body: JSON.stringify({ sports }),
  })
}

export function getRecentPredictions(limit = 100): Promise<RecentPrediction[]> {
  return request(`/predictions/recent?limit=${limit}&sort_by=newest`)
}

export function getLivePerformance(): Promise<PerformanceMap> {
  return request('/predictions/performance')
}

export function getModelPerformance(): Promise<ModelPerformance[]> {
  return request('/models/performance')
}

export function autoSettle(): Promise<SettleResult> {
  return request('/predictions/settle/auto', { method: 'POST' })
}

export function getLineMatrix(sport: string, home: string, away: string): Promise<LineMatrix> {
  return request(`/lines/matrix/${sport}/${encodeURIComponent(home)}/${encodeURIComponent(away)}`)
}

export function getRecentAlerts(limit = 50): Promise<AlertRecord[]> {
  return request(`/predictions/alerts/recent?limit=${limit}`)
}

export function getHealth(): Promise<HealthResponse> {
  return request('/health')
}
