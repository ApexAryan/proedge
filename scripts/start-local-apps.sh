#!/usr/bin/env bash
# Start KalshiEdge (:8000), ProEdge (:8010), and EDGE NBA (:8600) if not already running.
set -euo pipefail

KALSHI_ROOT="${KALSHI_ROOT:-$HOME/Desktop/kalshiedge/files}"
PROEDGE_ROOT="${PROEDGE_ROOT:-$HOME/Desktop/proedge}"
EDGE_ROOT="${EDGE_ROOT:-$HOME/Desktop/athlete database}"

port_taken() {
  lsof -ti :"$1" >/dev/null 2>&1
}

start_kalshi() {
  local port=8000
  if [[ -f "$KALSHI_ROOT/.env" ]]; then
    port=$(grep -E '^API_PORT=' "$KALSHI_ROOT/.env" | cut -d= -f2- | tr -d '\r' || true)
    port=${port:-8000}
  fi
  if port_taken "$port"; then
    echo "KalshiEdge  already running → http://localhost:$port"
    return
  fi
  if [[ ! -x "$KALSHI_ROOT/.venv/bin/python" ]]; then
    echo "KalshiEdge  missing venv at $KALSHI_ROOT/.venv — skip"
    return
  fi
  (cd "$KALSHI_ROOT" && nohup .venv/bin/python run_dev.py > /tmp/kalshiedge.log 2>&1 &)
  echo "KalshiEdge  starting on :$port → http://localhost:$port  (log: /tmp/kalshiedge.log)"
}

start_proedge() {
  local port=8010
  if [[ -f "$PROEDGE_ROOT/.env" ]]; then
    port=$(grep -E '^API_PORT=' "$PROEDGE_ROOT/.env" | cut -d= -f2- | tr -d '\r' || true)
    port=${port:-8010}
  fi
  if port_taken "$port"; then
    echo "ProEdge     already running → http://localhost:$port/dashboard"
    return
  fi
  if [[ ! -x "$PROEDGE_ROOT/.venv/bin/uvicorn" ]]; then
    echo "ProEdge     missing venv at $PROEDGE_ROOT/.venv — run: cd proedge && make install"
    return
  fi
  (cd "$PROEDGE_ROOT" && nohup .venv/bin/uvicorn proedge.api.main:app --host 0.0.0.0 --port "$port" --reload > /tmp/proedge.log 2>&1 &)
  echo "ProEdge     starting on :$port → http://localhost:$port/dashboard  (log: /tmp/proedge.log)"
}

start_edge_nba() {
  if port_taken 8600; then
    echo "EDGE NBA    already running → http://localhost:8600"
    return
  fi
  if [[ ! -x "$EDGE_ROOT/.venv/bin/uvicorn" ]]; then
    echo "EDGE NBA    missing venv at $EDGE_ROOT/.venv — skip"
    return
  fi
  (cd "$EDGE_ROOT" && nohup .venv/bin/uvicorn nba.api.main:app --host 127.0.0.1 --port 8600 > /tmp/edge-nba.log 2>&1 &)
  echo "EDGE NBA    starting on :8600 → http://localhost:8600  (log: /tmp/edge-nba.log)"
}

echo "Local sports apps"
start_kalshi
start_proedge
start_edge_nba
echo ""
echo "Run each in its own terminal instead: see README → Running alongside other local apps"
