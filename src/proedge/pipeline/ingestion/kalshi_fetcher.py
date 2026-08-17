"""Kalshi prediction market fetcher — game total implied lines.

Naming scheme (decoded from live API):
  NBA: KXNBATOTAL-{YYMONDD}{TEAM1}{TEAM2}-{THRESHOLD}
  MLB: KXMLBTOTAL-{YYMONDD}{HHMM}{TEAM1}{TEAM2}-{THRESHOLD}
  NFL: KXNFLTOTAL-{YYMONDD}{TEAM1}{TEAM2}-{THRESHOLD}  (offseason = no markets)

The yes_bid/yes_ask prices are in [0, 1] (dollars per $1 payout).
Midpoint ≈ probability that the total goes OVER the threshold.
We interpolate across thresholds to find the market-implied 50% line.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

_SERIES: dict[str, str] = {
    "nba": "KXNBATOTAL",
    "mlb": "KXMLBTOTAL",
    "nfl": "KXNFLTOTAL",
}

# Sports that embed a HHMM game time in the event ticker (multiple games per day)
_HAS_GAMETIME = {"mlb", "nfl"}


@dataclass
class KalshiThreshold:
    threshold: float
    yes_bid: float    # best bid to buy Over contract
    yes_ask: float    # best ask to buy Over contract (cost to bet Over)
    no_bid: float     # best bid to buy Under contract
    no_ask: float     # best ask to buy Under contract (cost to bet Under)
    implied_prob: float  # yes midpoint = market-implied P(over threshold)


@dataclass
class KalshiGameTotal:
    sport: str
    event_ticker: str       # e.g. KXNBATOTAL-26MAY11OKCLAL
    team1: str              # first team code parsed from ticker
    team2: str              # second team code parsed from ticker
    game_date: str          # YYMONDD string e.g. "26MAY11"
    implied_line: float     # interpolated 50%-probability total
    nearest_threshold: int  # nearest actual market threshold
    nearest_prob: float     # yes mid-price at nearest_threshold
    thresholds: list[tuple[int, float]] = field(default_factory=list)  # (thresh, yes_mid) sorted asc
    threshold_data: list[KalshiThreshold] = field(default_factory=list)  # full price data per threshold


def fetch_totals(sport: str, timeout: float = 10.0) -> list[KalshiGameTotal]:
    """Fetch all open game total markets for a sport and return implied lines.

    No authentication required — Kalshi market data is public.
    """
    series = _SERIES.get(sport.lower())
    if not series:
        return []

    params = {"status": "open", "limit": 1000, "series_ticker": series}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                f"{_BASE_URL}/markets",
                params=params,
                headers={"Accept": "application/json"},
            )
    except httpx.RequestError as exc:
        logger.warning("Kalshi: network error fetching %s: %s", sport, exc)
        return []

    if resp.status_code != 200:
        logger.warning("Kalshi: HTTP %d for series %s", resp.status_code, series)
        return []

    markets = resp.json().get("markets", [])
    logger.info("Kalshi: %d open %s total markets fetched", len(markets), sport.upper())
    return _parse_markets(sport.lower(), series, markets)


def get_implied_line(
    sport: str,
    home_team: str,
    away_team: str,
    timeout: float = 10.0,
) -> KalshiGameTotal | None:
    """Return the Kalshi-implied total for a specific matchup, or None if not found."""
    totals = fetch_totals(sport, timeout=timeout)
    home_up = home_team.upper()
    away_up = away_team.upper()

    for game in totals:
        ticker_up = game.event_ticker.upper()
        t1 = game.team1.upper()
        t2 = game.team2.upper()

        # Match either team order — Kalshi puts teams alphabetically
        home_hit = home_up in t1 or t1 in home_up or home_up in t2 or t2 in home_up
        away_hit = away_up in t1 or t1 in away_up or away_up in t2 or t2 in away_up

        # Fallback: both codes appear anywhere in the ticker
        if not (home_hit and away_hit):
            home_hit = home_up in ticker_up
            away_hit = away_up in ticker_up

        if home_hit and away_hit:
            return game

    return None


# ── Internal helpers ──────────────────────────────────────────────────────────


def _parse_markets(
    sport: str, series: str, markets: list[dict]
) -> list[KalshiGameTotal]:
    events: dict[str, list[tuple[int, float]]] = {}

    # event_ticker → list of (threshold, yes_bid, yes_ask, no_bid, no_ask)
    events: dict[str, list[tuple[int, float, float, float, float]]] = {}

    for m in markets:
        ticker: str = m.get("ticker", "")
        parts = ticker.rsplit("-", 1)
        if len(parts) != 2:
            continue
        event_ticker, thresh_str = parts
        try:
            threshold = int(thresh_str)
        except ValueError:
            # Try float threshold (e.g. half-point lines stored as int after strip)
            try:
                threshold = int(float(thresh_str))
            except ValueError:
                continue

        yes_bid = float(m.get("yes_bid_dollars") or 0)
        yes_ask = float(m.get("yes_ask_dollars") or 0)
        no_bid = float(m.get("no_bid_dollars") or 0)
        no_ask = float(m.get("no_ask_dollars") or 0)

        events.setdefault(event_ticker, []).append((threshold, yes_bid, yes_ask, no_bid, no_ask))

    results: list[KalshiGameTotal] = []
    for event_ticker, raw_pts in events.items():
        raw_sorted = sorted(raw_pts, key=lambda x: x[0])  # ascending threshold

        # Build KalshiThreshold list
        td_list: list[KalshiThreshold] = []
        pts_sorted: list[tuple[int, float]] = []
        for thresh, yb, ya, nb, na in raw_sorted:
            yes_mid = round((yb + ya) / 2, 4)
            # Kalshi ticker integer N means "Over N" (strictly), which equals
            # conventional sportsbook notation of N + 0.5 (no tie possible).
            display_thresh = float(thresh) + 0.5
            td_list.append(KalshiThreshold(
                threshold=display_thresh,
                yes_bid=round(yb, 4),
                yes_ask=round(ya, 4),
                no_bid=round(nb, 4),
                no_ask=round(na, 4),
                implied_prob=yes_mid,
            ))
            pts_sorted.append((thresh, yes_mid))

        suffix = event_ticker[len(series) + 1:]  # strip "KXNBATOTAL-"
        game_date, team1, team2 = _parse_event_suffix(sport, suffix)
        implied = round(_interpolate_50pct(pts_sorted) + 0.5, 2)  # same +0.5 adjustment
        nearest = min(pts_sorted, key=lambda x: abs(x[1] - 0.5))

        results.append(
            KalshiGameTotal(
                sport=sport,
                event_ticker=event_ticker,
                team1=team1,
                team2=team2,
                game_date=game_date,
                implied_line=implied,
                nearest_threshold=nearest[0],
                nearest_prob=round(nearest[1], 4),
                thresholds=pts_sorted,
                threshold_data=td_list,
            )
        )

    logger.info("Kalshi: parsed %d %s games", len(results), sport.upper())
    return results


def _parse_event_suffix(sport: str, suffix: str) -> tuple[str, str, str]:
    """Parse YYMONDD[HHMM]TEAM1TEAM2 → (date_str, team1, team2)."""
    date_m = re.match(r"^(\d{2}[A-Z]{3}\d{2})", suffix)
    if not date_m:
        return suffix, "UNK", "UNK"

    game_date = date_m.group(1)
    rest = suffix[len(game_date):]

    # MLB and NFL embed a 4-digit HHMM game time after the date
    if sport in _HAS_GAMETIME and rest and rest[0].isdigit():
        rest = rest[4:]

    team1, team2 = _split_teams(rest)
    return game_date, team1, team2


# Teams whose Kalshi ticker abbreviation is 2 letters (not the standard 3).
# Affects how 5-char concatenated codes are split (2+3 vs 3+2).
_TWO_LETTER_CODES = {"SF", "AZ", "TB", "KC", "SD", "NY", "LA"}


def _split_teams(codes: str) -> tuple[str, str]:
    """Split concatenated team codes. Most are 3+3 chars; some are 2+3 or 3+2."""
    n = len(codes)
    if n == 6:
        return codes[:3], codes[3:]
    if n == 5:
        # If the first 2 chars are a known 2-letter code, split 2+3 (e.g. SF+LAD, AZ+TEX, TB+TOR)
        if codes[:2].upper() in _TWO_LETTER_CODES:
            return codes[:2], codes[2:]
        return codes[:3], codes[3:]
    if n == 4:
        return codes[:2], codes[2:]
    return codes, "UNK"


def _interpolate_50pct(pts: list[tuple[int, float]]) -> float:
    """Linearly interpolate to find the threshold where yes_mid ≈ 0.50.

    pts is sorted by threshold ascending; yes_mid decreases as threshold rises.
    """
    if not pts:
        return 0.0

    above = [(t, p) for t, p in pts if p >= 0.50]
    below = [(t, p) for t, p in pts if p < 0.50]

    if not above:
        return float(pts[-1][0])  # all below 50% — return highest threshold
    if not below:
        return float(pts[0][0])   # all above 50% — return lowest threshold

    # Highest threshold still at/above 50%, and lowest threshold below 50%
    t_hi, p_hi = max(above, key=lambda x: x[0])
    t_lo, p_lo = min(below, key=lambda x: x[0])

    if p_hi == p_lo:
        return float(t_hi)

    # Linear interpolation: find t where prob = 0.50
    frac = (p_hi - 0.50) / (p_hi - p_lo)
    return round(t_hi + frac * (t_lo - t_hi), 2)
