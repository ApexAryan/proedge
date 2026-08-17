export type Sport = 'nba' | 'mlb' | 'nfl'
export type Side = 'over' | 'under'

export interface KalshiThreshold {
  threshold: number
  yes_ask: number
  no_ask: number
  market_prob: number
  model_prob: number | null
  ev_yes: number | null
  ev_no: number | null
  best_bet: Side | null
  edge: number | null
}

export interface MarketScanGame {
  game_id: string | null
  prediction_id: string | null
  sport: Sport
  home_team: string
  away_team: string
  kalshi_implied_line: number
  model_prob_over: number
  model_prob_under: number
  predicted_direction: Side
  confidence: number
  best_threshold: number | null
  best_bet: Side | null
  best_edge: number | null
  thresholds: KalshiThreshold[]
  line_movement: number | null
  ci_lower: number | null
  ci_upper: number | null
  home_key_players_out: number
  away_key_players_out: number
  live_stats: boolean
}

export interface MarketScanResponse {
  scanned_at: string
  sports: string[]
  total_games: number
  results: MarketScanGame[]
}

export interface SportPerformance {
  total_settled: number
  wins: number
  losses: number
  hit_rate: number | null
  pnl: number | null
}

export interface OverallPerformance {
  logged: number
  pending: number
  total_settled: number
  wins: number
  losses: number
  hit_rate: number | null
  roi: number | null
  pnl: number | null
}

export type PerformanceMap = Record<Sport, SportPerformance> & {
  overall?: OverallPerformance
}

export interface ModelPerformance {
  version: string
  sport: string
  accuracy: number | null
  trained_at: string
  is_active: boolean
}

export interface RecentPrediction {
  prediction_id: string
  game_id: string
  sport: Sport
  home_team: string
  away_team: string
  game_date: string | null
  total_line: number | null
  model_version: string
  prob_over: number
  prob_under: number
  ci_lower: number
  ci_upper: number
  predicted_direction: Side
  confidence: number
  predicted_at: string | null
  is_correct: boolean | null
  clv: number | null
  actual_total: number | null
  closing_line: number | null
  settled_at: string | null
}

export interface LineMatrix {
  sport: string
  home_team: string
  away_team: string
  kalshi_implied_line: number | null
  prizepicks_line: number | null
  underdog_line: number | null
  underdog_over_american: string | null
  underdog_under_american: string | null
  thresholds: KalshiThreshold[]
  fetched_at: string
}

export interface AlertRecord {
  alert_id: string
  sport: Sport
  home_team: string
  away_team: string
  game_date: string
  direction: string
  confidence: number
  prob_over: number
  edge: number | null
  total_line: number | null
  fired: boolean
  created_at: string
}

export interface HealthResponse {
  status: string
  db_connected: boolean
  models_loaded: Record<string, string | null>
  uptime_seconds: number
  version: string
}

export interface SettleResult {
  settled?: number
  already_settled?: number
  no_score_match?: number
  error?: string
}
