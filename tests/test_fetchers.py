"""Unit tests for all data fetchers — HTTP calls are mocked."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# OddsFetcher
# ============================================================================

class TestOddsFetcherSportKey:
    def test_known_sports(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        f = OddsFetcher(api_key="key")
        assert f._sport_key("nba") == "basketball_nba"
        assert f._sport_key("nfl") == "americanfootball_nfl"
        assert f._sport_key("mlb") == "baseball_mlb"

    def test_case_insensitive(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        assert OddsFetcher(api_key="x")._sport_key("NBA") == "basketball_nba"

    def test_unknown_sport_raises(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        with pytest.raises(ValueError, match="Unknown sport"):
            OddsFetcher(api_key="x")._sport_key("hockey")


class TestOddsFetcherParseEvent:
    def _event(self, **overrides):
        base = {
            "id": "evt-001",
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
            "commence_time": "2026-05-10T02:00:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 228.5},
                                {"name": "Under", "point": 228.5},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Los Angeles Lakers", "point": -5.5},
                                {"name": "Boston Celtics", "point": 5.5},
                            ],
                        },
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Los Angeles Lakers", "price": -220},
                                {"name": "Boston Celtics", "price": 180},
                            ],
                        },
                    ],
                }
            ],
        }
        base.update(overrides)
        return base

    def test_parse_returns_game_odds(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher, GameOdds
        fetcher = OddsFetcher(api_key="key")
        result = fetcher._parse_event(self._event(), "nba")
        assert isinstance(result, GameOdds)

    def test_total_line_parsed(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        result = OddsFetcher(api_key="key")._parse_event(self._event(), "nba")
        assert result.total_line == 228.5

    def test_spread_parsed(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        result = OddsFetcher(api_key="key")._parse_event(self._event(), "nba")
        assert result.spread == -5.5

    def test_moneyline_parsed(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        result = OddsFetcher(api_key="key")._parse_event(self._event(), "nba")
        assert result.home_ml == -220
        assert result.away_ml == 180

    def test_teams_preserved(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        result = OddsFetcher(api_key="key")._parse_event(self._event(), "nba")
        assert result.home_team == "Los Angeles Lakers"
        assert result.away_team == "Boston Celtics"

    def test_bookmaker_count(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        result = OddsFetcher(api_key="key")._parse_event(self._event(), "nba")
        assert result.bookmaker_count == 1
        assert "draftkings" in result.sources

    def test_empty_bookmakers_gives_none_totals(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        result = OddsFetcher(api_key="key")._parse_event(
            self._event(bookmakers=[]), "nba"
        )
        assert result.total_line is None
        assert result.spread is None

    def test_bad_commence_time_falls_back_to_now(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        result = OddsFetcher(api_key="key")._parse_event(
            self._event(commence_time="not-a-date"), "nba"
        )
        assert isinstance(result.commence_time, datetime)

    def test_consensus_median_across_bookmakers(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        event = self._event()
        # Add a second bookmaker with a different total
        event["bookmakers"].append({
            "key": "fanduel",
            "markets": [
                {"key": "totals", "outcomes": [{"name": "Over", "point": 229.5}]},
            ],
        })
        result = OddsFetcher(api_key="key")._parse_event(event, "nba")
        assert result.total_line == pytest.approx(229.0)  # median of [228.5, 229.5]


class TestOddsFetcherFetchGameOdds:
    def _mock_resp(self, status: int, body=None, headers=None):
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = body or []
        resp.headers = headers or {}
        resp.text = ""
        return resp

    def test_empty_api_key_returns_empty(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        assert OddsFetcher(api_key="").fetch_game_odds("nba") == []

    def test_401_returns_empty(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                self._mock_resp(401)
            )
            assert OddsFetcher(api_key="bad").fetch_game_odds("nba") == []

    def test_429_returns_empty(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                self._mock_resp(429)
            )
            assert OddsFetcher(api_key="key").fetch_game_odds("nba") == []

    def test_500_returns_empty(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                self._mock_resp(500)
            )
            assert OddsFetcher(api_key="key").fetch_game_odds("nba") == []

    def test_network_error_returns_empty(self):
        import httpx
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = (
                httpx.ConnectError("unreachable")
            )
            assert OddsFetcher(api_key="key").fetch_game_odds("nba") == []

    def test_200_returns_game_odds_list(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher, GameOdds
        payload = [
            {
                "id": "g1",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "commence_time": "2026-05-10T02:00:00Z",
                "bookmakers": [],
            }
        ]
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = (
                self._mock_resp(200, body=payload, headers={"x-requests-remaining": "450"})
            )
            result = OddsFetcher(api_key="key").fetch_game_odds("nba")
        assert len(result) == 1
        assert isinstance(result[0], GameOdds)


class TestOddsFetcherGetTotalLine:
    def test_partial_match_returns_line(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher, GameOdds
        game = GameOdds(
            game_id="g1",
            sport="nba",
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            commence_time=datetime.now(timezone.utc),
            total_line=228.5,
            spread=-5.5,
            home_ml=-220,
            away_ml=180,
            bookmaker_count=1,
        )
        fetcher = OddsFetcher(api_key="key")
        with patch.object(fetcher, "fetch_game_odds", return_value=[game]):
            line = fetcher.get_total_line("nba", "Lakers", "Celtics")
        assert line == 228.5

    def test_no_match_returns_none(self):
        from proedge.pipeline.ingestion.odds_fetcher import OddsFetcher
        fetcher = OddsFetcher(api_key="key")
        with patch.object(fetcher, "fetch_game_odds", return_value=[]):
            assert fetcher.get_total_line("nba", "MIL", "PHX") is None


# ============================================================================
# ESPN NFL Fetcher — pure functions
# ============================================================================

class TestESPNParsers:
    def test_parse_stat_found(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _parse_stat
        stats = [{"label": "Passing", "displayValue": "312"}, {"label": "Rushing", "displayValue": "95"}]
        assert _parse_stat(stats, "Passing") == "312"

    def test_parse_stat_not_found(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _parse_stat
        assert _parse_stat([], "Passing") is None

    def test_parse_fraction_valid(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _parse_fraction
        assert _parse_fraction("7-12") == pytest.approx(7 / 12)

    def test_parse_fraction_zero_denominator(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _parse_fraction
        assert _parse_fraction("0-0") == 0.0

    def test_parse_fraction_none(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _parse_fraction
        assert _parse_fraction(None) == 0.0

    def test_parse_time_of_possession_valid(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _parse_time_of_possession
        assert _parse_time_of_possession("31:24") == pytest.approx(31 + 24 / 60)

    def test_parse_time_of_possession_none(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _parse_time_of_possession
        assert _parse_time_of_possession(None) == 30.0

    def test_parse_team_stats_returns_all_keys(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _parse_team_stats
        from proedge.pipeline.ingestion.stats import STAT_KEYS
        stats = [
            {"label": "Passing", "displayValue": "285"},
            {"label": "Rushing", "displayValue": "110"},
            {"label": "Total Yards", "displayValue": "395"},
            {"label": "Turnovers", "displayValue": "1"},
            {"label": "Sacks-Yards Lost", "displayValue": "3-22"},
            {"label": "3rd down efficiency", "displayValue": "6-14"},
            {"label": "Red Zone (Made-Att)", "displayValue": "2-3"},
            {"label": "Possession", "displayValue": "31:20"},
            {"label": "Yards per Play", "displayValue": "5.8"},
            {"label": "4th down efficiency", "displayValue": "1-2"},
            {"label": "Penalties", "displayValue": "6-55"},
            {"label": "Total Plays", "displayValue": "68"},
            {"label": "Comp/Att", "displayValue": "22/35"},
        ]
        result = _parse_team_stats(stats)
        for key in STAT_KEYS["nfl"]:
            assert key in result, f"Missing stat: {key}"

    def test_count_injuries(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _count_injuries
        comp = {"injuries": [{"status": "Out"}, {"status": "Doubtful"}, {"status": "Questionable"}]}
        assert _count_injuries(comp) == 2  # Out + Doubtful only


class TestESPNBuildGameRow:
    def _event(self, home="KC", away="SF", home_score=27, away_score=21):
        return {
            "id": "nfl-001",
            "date": "2023-11-05T17:00:00Z",
            "competitions": [{
                "status": {"type": {"name": "STATUS_FINAL"}},
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"abbreviation": home},
                        "score": str(home_score),
                        "injuries": [],
                    },
                    {
                        "homeAway": "away",
                        "team": {"abbreviation": away},
                        "score": str(away_score),
                        "injuries": [{"status": "Out"}],
                    },
                ],
                "weather": {"temperature": 55, "windSpeed": 12},
                "venue": {"fullName": "Arrowhead Stadium"},
            }],
        }

    def _summary(self, home="KC", away="SF"):
        def _stats(passing=300, rushing=100):
            return [
                {"label": "Passing", "displayValue": str(passing)},
                {"label": "Rushing", "displayValue": str(rushing)},
                {"label": "Total Yards", "displayValue": str(passing + rushing)},
                {"label": "Turnovers", "displayValue": "1"},
                {"label": "Sacks-Yards Lost", "displayValue": "2-15"},
                {"label": "3rd down efficiency", "displayValue": "7-13"},
                {"label": "Red Zone (Made-Att)", "displayValue": "3-4"},
                {"label": "Possession", "displayValue": "30:00"},
                {"label": "Yards per Play", "displayValue": "5.5"},
                {"label": "4th down efficiency", "displayValue": "0-1"},
                {"label": "Penalties", "displayValue": "5-45"},
                {"label": "Total Plays", "displayValue": "65"},
                {"label": "Comp/Att", "displayValue": "20/32"},
            ]
        return {
            "boxscore": {
                "teams": [
                    {"team": {"abbreviation": home}, "statistics": _stats(300, 120)},
                    {"team": {"abbreviation": away}, "statistics": _stats(250, 80)},
                ]
            }
        }

    def test_builds_valid_row(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _build_game_row
        row = _build_game_row(self._event(), self._summary(), 2023)
        assert row is not None
        assert row["home_team"] == "KC"
        assert row["away_team"] == "SF"
        assert row["home_score"] == 27
        assert row["away_score"] == 21
        assert row["total"] == 48
        assert row["sport"] == "nfl"

    def test_weather_fields(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _build_game_row
        row = _build_game_row(self._event(), self._summary(), 2023)
        assert row["wind_speed_mph"] == 12.0
        assert row["temperature_f"] == 55.0

    def test_dome_team_zeroes_wind(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _build_game_row
        # ATL is a dome team
        row = _build_game_row(self._event(home="ATL", away="SF"), self._summary("ATL", "SF"), 2023)
        assert row["wind_speed_mph"] == 0.0
        assert row["is_dome"] == 1.0

    def test_injury_count_from_competitors(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _build_game_row
        row = _build_game_row(self._event(), self._summary(), 2023)
        assert row["away_key_players_out"] == 1.0
        assert row["home_key_players_out"] == 0.0

    def test_non_final_returns_none(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _build_game_row
        event = self._event()
        event["competitions"][0]["status"]["type"]["name"] = "STATUS_IN_PROGRESS"
        assert _build_game_row(event, self._summary(), 2023) is None

    def test_altitude_denver(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _build_game_row
        row = _build_game_row(self._event(home="DEN", away="KC"), self._summary("DEN", "KC"), 2023)
        assert row["altitude_feet"] == 5280.0


class TestESPNFetchScoreboard:
    def test_returns_events_on_success(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _fetch_scoreboard
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"events": [{"id": "1"}, {"id": "2"}]}
        mock_client.get.return_value = mock_resp
        events = _fetch_scoreboard(mock_client, 2023, 5)
        assert len(events) == 2

    def test_returns_empty_on_error(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _fetch_scoreboard
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("timeout")
        assert _fetch_scoreboard(mock_client, 2023, 5) == []

    def test_returns_empty_when_no_events_key(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _fetch_scoreboard
        mock_client = MagicMock()
        mock_client.get.return_value.json.return_value = {}
        assert _fetch_scoreboard(mock_client, 2023, 5) == []


class TestESPNFetchSummary:
    def test_returns_dict_on_success(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _fetch_summary
        mock_client = MagicMock()
        mock_client.get.return_value.json.return_value = {"boxscore": {}}
        result = _fetch_summary(mock_client, "game-1")
        assert "boxscore" in result

    def test_returns_empty_dict_on_error(self):
        from proedge.pipeline.ingestion.espn_nfl_fetcher import _fetch_summary
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("timeout")
        assert _fetch_summary(mock_client, "game-1") == {}


# ============================================================================
# MLB Stats Fetcher — pure functions
# ============================================================================

class TestMLBParsers:
    def _side(self):
        return {
            "teamStats": {
                "batting": {
                    "runs": 5, "hits": 10, "homeRuns": 2, "strikeOuts": 8,
                    "baseOnBalls": 3, "avg": ".265", "obp": ".330",
                    "slg": ".440", "ops": ".770",
                },
                "pitching": {
                    "runs": 4, "strikeOuts": 9, "baseOnBalls": 2,
                    "era": "3.85", "whip": "1.20",
                    "flyOuts": 15, "groundOuts": 12,
                },
            },
            "errors": 1,
            "players": {},
        }

    def test_parse_side_stats_keys(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _parse_side_stats
        from proedge.pipeline.ingestion.stats import STAT_KEYS
        result = _parse_side_stats(self._side())
        for key in STAT_KEYS["mlb"]:
            assert key in result, f"Missing stat: {key}"

    def test_parse_side_stats_values(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _parse_side_stats
        result = _parse_side_stats(self._side())
        assert result["runsScored"] == 5.0
        assert result["hits"] == 10.0
        assert result["homeRuns"] == 2.0

    def test_count_il_players_none_on_il(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _count_il_players
        side = {"players": {"p1": {"status": {"code": "A"}}, "p2": {"status": {"code": "NRI"}}}}
        assert _count_il_players(side) == 0

    def test_count_il_players_detects_il(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _count_il_players
        side = {
            "players": {
                "p1": {"status": {"code": "IL"}},
                "p2": {"status": {"code": "DL"}},
                "p3": {"status": {"code": "A"}},
            }
        }
        assert _count_il_players(side) == 2


class TestMLBFetchTeamMap:
    def test_returns_id_to_abbr_dict(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _fetch_team_map
        payload = {"teams": [{"id": 119, "abbreviation": "LAD"}, {"id": 111, "abbreviation": "BOS"}]}
        mock_client = MagicMock()
        mock_client.get.return_value.json.return_value = payload
        result = _fetch_team_map(mock_client)
        assert result[119] == "LAD"
        assert result[111] == "BOS"

    def test_returns_empty_on_error(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _fetch_team_map
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("network")
        assert _fetch_team_map(mock_client) == {}


class TestMLBFetchSchedule:
    def test_returns_final_games_only(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _fetch_schedule
        payload = {
            "dates": [{
                "games": [
                    {"gamePk": 1, "status": {"detailedState": "Final"}},
                    {"gamePk": 2, "status": {"detailedState": "In Progress"}},
                ]
            }]
        }
        mock_client = MagicMock()
        mock_client.get.return_value.json.return_value = payload
        games = _fetch_schedule(mock_client, "2023-04-01", "2023-04-01")
        assert len(games) == 1
        assert games[0]["gamePk"] == 1

    def test_returns_empty_on_error(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _fetch_schedule
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("timeout")
        assert _fetch_schedule(mock_client, "2023-04-01", "2023-04-01") == []


class TestMLBBuildGameRow:
    def _game(self):
        return {
            "gamePk": 717465,
            "gameDate": "2023-05-01T17:05:00Z",
            "officialDate": "2023-05-01",
            "teams": {
                "home": {"team": {"id": 119}, "score": 5},
                "away": {"team": {"id": 111}, "score": 3},
            },
            "weather": {"temp": "72", "wind": "8 mph, Out to CF"},
        }

    def _boxscore(self):
        side = {
            "teamStats": {
                "batting": {
                    "runs": 5, "hits": 9, "homeRuns": 1, "strikeOuts": 7,
                    "baseOnBalls": 3, "avg": ".260", "obp": ".320",
                    "slg": ".400", "ops": ".720",
                },
                "pitching": {
                    "runs": 3, "strikeOuts": 8, "baseOnBalls": 2,
                    "era": "3.50", "whip": "1.15", "flyOuts": 12, "groundOuts": 10,
                },
            },
            "errors": 0,
            "players": {},
        }
        return {"teams": {"home": side, "away": side}}

    def test_builds_valid_row(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _build_game_row
        team_map = {119: "LAD", 111: "BOS"}
        row = _build_game_row(self._game(), self._boxscore(), team_map, 2023)
        assert row is not None
        assert row["home_team"] == "LAD"
        assert row["away_team"] == "BOS"
        assert row["home_score"] == 5
        assert row["away_score"] == 3
        assert row["total"] == 8
        assert row["sport"] == "mlb"

    def test_weather_fields(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _build_game_row
        team_map = {119: "LAD", 111: "BOS"}
        row = _build_game_row(self._game(), self._boxscore(), team_map, 2023)
        assert row["wind_speed_mph"] == 8.0
        assert row["temperature_f"] == 72.0

    def test_coors_field_altitude(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _build_game_row
        game = self._game()
        game["teams"]["home"]["team"]["id"] = 115  # COL team id (use team_map)
        team_map = {115: "COL", 111: "BOS"}
        row = _build_game_row(game, self._boxscore(), team_map, 2023)
        assert row["altitude_feet"] == 5280.0

    def test_dome_team_zeroes_wind(self):
        from proedge.pipeline.ingestion.mlb_stats_fetcher import _build_game_row
        game = self._game()
        game["teams"]["home"]["team"]["id"] = 139  # TB
        team_map = {139: "TB", 111: "BOS"}
        row = _build_game_row(game, self._boxscore(), team_map, 2023)
        assert row["is_dome"] == 1.0
        assert row["wind_speed_mph"] == 0.0


# ============================================================================
# NBA Fetcher — pure functions (no nba_api network calls)
# ============================================================================

class TestNBASeasonYear:
    def test_october_is_current_year(self):
        from proedge.pipeline.ingestion.nba_fetcher import _season_year
        ts = pd.Timestamp("2023-10-25")
        assert _season_year(ts) == 2023

    def test_january_is_previous_year(self):
        from proedge.pipeline.ingestion.nba_fetcher import _season_year
        ts = pd.Timestamp("2024-01-15")
        assert _season_year(ts) == 2023

    def test_september_is_previous_year(self):
        from proedge.pipeline.ingestion.nba_fetcher import _season_year
        ts = pd.Timestamp("2024-09-30")
        assert _season_year(ts) == 2023


class TestNBAComputeProxyLines:
    def _small_df(self, n: int = 30) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        teams = ["BOS", "LAL", "GSW", "MIA"]
        rows = []
        for i in range(n):
            h = teams[i % len(teams)]
            a = teams[(i + 1) % len(teams)]
            h_pts = int(rng.normal(112, 8))
            a_pts = int(rng.normal(108, 8))
            rows.append({
                "game_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "home_team": h, "away_team": a,
                "home_score": h_pts, "away_score": a_pts,
                "total": h_pts + a_pts,
                "total_line": float(h_pts + a_pts) + float(rng.normal(0, 2)),
            })
        return pd.DataFrame(rows)

    def test_returns_dataframe_with_total_line(self):
        from proedge.pipeline.ingestion.nba_fetcher import _compute_proxy_lines
        df = self._small_df()
        result = _compute_proxy_lines(df)
        assert "total_line" in result.columns
        assert "result_over" in result.columns

    def test_total_lines_in_realistic_range(self):
        from proedge.pipeline.ingestion.nba_fetcher import _compute_proxy_lines
        result = _compute_proxy_lines(self._small_df())
        assert result["total_line"].between(180, 280).all()

    def test_result_over_is_binary(self):
        from proedge.pipeline.ingestion.nba_fetcher import _compute_proxy_lines
        result = _compute_proxy_lines(self._small_df())
        assert set(result["result_over"].unique()).issubset({0, 1})

    def test_original_df_not_mutated(self):
        from proedge.pipeline.ingestion.nba_fetcher import _compute_proxy_lines
        df = self._small_df()
        cols_before = set(df.columns)
        _compute_proxy_lines(df)
        assert set(df.columns) == cols_before  # no columns added in-place
