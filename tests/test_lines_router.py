"""Tests for GET /lines/prizepicks endpoints."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from proedge.api.main import app
from proedge.pipeline.ingestion.prizepicks_fetcher import (
    GameLine,
    PlayerProjection,
    PrizePicksBoard,
)

client = TestClient(app)


def _empty_board(sport: str = "nba") -> PrizePicksBoard:
    return PrizePicksBoard(sport=sport, fetched_at=datetime.now(timezone.utc))


def _proj(**overrides) -> PlayerProjection:
    defaults = dict(
        projection_id="proj-1",
        player_name="Player A",
        team="BOS",
        position="G",
        stat_type="Points",
        line=22.5,
        game_id="game-1",
        home_team="BOS",
        away_team="LAL",
        start_time=datetime(2026, 5, 10, 20, 0, tzinfo=timezone.utc),
        status="pre_game",
        is_promo=False,
        odds_type="standard",
        projection_type="Single Stat",
        sport="nba",
    )
    return PlayerProjection(**{**defaults, **overrides})


def _game_line(**overrides) -> GameLine:
    defaults = dict(
        game_id="game-1",
        home_team="BOS",
        away_team="LAL",
        start_time=datetime(2026, 5, 10, 20, 0, tzinfo=timezone.utc),
        stat_type="Total Points",
        line=228.5,
        sport="nba",
    )
    return GameLine(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# GET /lines/prizepicks/{sport}
# ---------------------------------------------------------------------------

def test_lines_invalid_sport_returns_422():
    resp = client.get("/lines/prizepicks/hockey")
    assert resp.status_code == 422


def test_lines_board_empty():
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=_empty_board())):
        resp = client.get("/lines/prizepicks/nba")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sport"] == "nba"
    assert data["game_count"] == 0
    assert data["player_prop_count"] == 0
    assert data["game_line_count"] == 0
    assert data["games"] == []


def test_lines_board_response_structure():
    board = PrizePicksBoard(
        sport="nba",
        fetched_at=datetime.now(timezone.utc),
        player_projections=[_proj()],
        game_lines=[_game_line()],
    )
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=board)):
        resp = client.get("/lines/prizepicks/nba")
    assert resp.status_code == 200
    data = resp.json()
    assert data["game_count"] == 1
    assert data["player_prop_count"] == 1
    assert data["game_line_count"] == 1
    for key in ("sport", "fetched_at", "game_count", "player_prop_count",
                "game_line_count", "games"):
        assert key in data, f"Missing key: {key}"


def test_lines_board_game_summary_structure():
    board = PrizePicksBoard(
        sport="nba",
        fetched_at=datetime.now(timezone.utc),
        player_projections=[_proj()],
        game_lines=[_game_line()],
    )
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=board)):
        resp = client.get("/lines/prizepicks/nba")
    game = resp.json()["games"][0]
    for key in ("game_id", "home_team", "away_team", "sport",
                "total_line", "spread", "game_lines", "player_projections"):
        assert key in game, f"Missing game key: {key}"


def test_lines_board_filters_promos_by_default():
    promo = _proj(is_promo=True)
    regular = _proj(projection_id="proj-2", is_promo=False)
    board = PrizePicksBoard(
        sport="nba",
        fetched_at=datetime.now(timezone.utc),
        player_projections=[promo, regular],
        game_lines=[],
    )
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=board)):
        resp = client.get("/lines/prizepicks/nba?include_promos=false")
    assert resp.status_code == 200
    total_props = sum(len(g["player_projections"]) for g in resp.json()["games"])
    assert total_props == 1  # promo filtered; regular kept


def test_lines_board_include_promos_true():
    promo = _proj(is_promo=True)
    regular = _proj(projection_id="proj-2", is_promo=False)
    board = PrizePicksBoard(
        sport="nba",
        fetched_at=datetime.now(timezone.utc),
        player_projections=[promo, regular],
        game_lines=[],
    )
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=board)):
        resp = client.get("/lines/prizepicks/nba?include_promos=true")
    total_props = sum(len(g["player_projections"]) for g in resp.json()["games"])
    assert total_props == 2


def test_lines_board_status_filter():
    pre = _proj(status="pre_game")
    locked = _proj(projection_id="proj-2", status="locked")
    board = PrizePicksBoard(
        sport="nba",
        fetched_at=datetime.now(timezone.utc),
        player_projections=[pre, locked],
        game_lines=[],
    )
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=board)):
        resp = client.get("/lines/prizepicks/nba?status_filter=pre_game")
    total_props = sum(len(g["player_projections"]) for g in resp.json()["games"])
    assert total_props == 1  # only pre_game kept


def test_lines_board_prizepicks_http_error_returns_502():
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = "Too Many Requests"
    with patch(
        "proedge.api.routers.lines.fetch_board",
        new=AsyncMock(side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=mock_response)),
    ):
        resp = client.get("/lines/prizepicks/nba")
    assert resp.status_code == 502


def test_lines_board_prizepicks_network_error_returns_503():
    with patch(
        "proedge.api.routers.lines.fetch_board",
        new=AsyncMock(side_effect=httpx.ConnectError("unreachable")),
    ):
        resp = client.get("/lines/prizepicks/nba")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /lines/prizepicks/{sport}/game/{home}/{away}
# ---------------------------------------------------------------------------

def test_lines_game_not_found_returns_404():
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=_empty_board())):
        resp = client.get("/lines/prizepicks/nba/game/BOS/LAL")
    assert resp.status_code == 404


def test_lines_game_found_returns_summary():
    board = PrizePicksBoard(
        sport="nba",
        fetched_at=datetime.now(timezone.utc),
        player_projections=[_proj()],
        game_lines=[_game_line()],
    )
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=board)):
        resp = client.get("/lines/prizepicks/nba/game/BOS/LAL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["home_team"] == "BOS"
    assert data["away_team"] == "LAL"
    assert data["total_line"] == 228.5


def test_lines_game_case_insensitive_team_match():
    board = PrizePicksBoard(
        sport="nba",
        fetched_at=datetime.now(timezone.utc),
        player_projections=[_proj()],
        game_lines=[],
    )
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=board)):
        resp = client.get("/lines/prizepicks/nba/game/bos/lal")
    assert resp.status_code == 200


def test_lines_game_invalid_sport_returns_422():
    resp = client.get("/lines/prizepicks/hockey/game/BOS/LAL")
    assert resp.status_code == 422


def test_lines_game_only_props_no_explicit_total_line():
    """When no game-level line exists, projected_total can be derived from props."""
    props = [_proj(player_name=f"Player {i}", projection_id=f"proj-{i}") for i in range(8)]
    board = PrizePicksBoard(
        sport="nba",
        fetched_at=datetime.now(timezone.utc),
        player_projections=props,
        game_lines=[],
    )
    with patch("proedge.api.routers.lines.fetch_board", new=AsyncMock(return_value=board)):
        resp = client.get("/lines/prizepicks/nba/game/BOS/LAL")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_line"] is None
    # projected_total can be None or a float — just verify it doesn't crash
    assert "projected_total" in data
