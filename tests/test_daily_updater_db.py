"""
DB-layer tests for DailyUpdater._persist_injury_reports() and ._settle_predictions().

All database I/O is mocked via SyncSessionLocal — no real Postgres needed.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pandas as pd

from proedge.pipeline.ingestion.daily_updater import DailyUpdater
from proedge.pipeline.ingestion.injuries import InjuredPlayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _updater(sport: str = "nba") -> DailyUpdater:
    return DailyUpdater(sport, data_dir="/tmp/proedge_test_updater")


def _player(name: str = "LeBron James", is_key: bool = True, comment: str = "Knee") -> InjuredPlayer:
    return InjuredPlayer(name=name, team="LAL", status="Out", is_key=is_key, comment=comment)


def _mock_session() -> MagicMock:
    """Return a MagicMock that behaves as a context manager for SyncSessionLocal."""
    s = MagicMock()
    s.__enter__ = MagicMock(return_value=s)
    s.__exit__ = MagicMock(return_value=False)
    return s


def _games_df(**overrides) -> pd.DataFrame:
    defaults = {
        "game_id": ["0021234567"],
        "home_score": [112],
        "away_score": [108],
        "total": [220],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


# ---------------------------------------------------------------------------
# _persist_injury_reports()
# ---------------------------------------------------------------------------

class TestPersistInjuryReports:

    def test_empty_records_skips_db(self):
        updater = _updater()
        with patch("proedge.db.session.SyncSessionLocal") as mock_cls:
            updater._persist_injury_reports([])
        mock_cls.assert_not_called()

    def test_all_empty_player_lists_skips_db(self):
        updater = _updater()
        records = [("LAL", "nba", []), ("BOS", "nba", [])]
        with patch("proedge.db.session.SyncSessionLocal") as mock_cls:
            updater._persist_injury_reports(records)
        mock_cls.assert_not_called()

    def test_key_player_gets_status_out(self):
        updater = _updater()
        added = []
        session = _mock_session()
        session.add_all.side_effect = lambda items: added.extend(items)

        records = [("LAL", "nba", [_player(is_key=True)])]
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._persist_injury_reports(records)

        assert len(added) == 1
        assert added[0].status == "out"
        assert added[0].impact_score == 1.0

    def test_non_key_player_gets_status_questionable(self):
        updater = _updater()
        added = []
        session = _mock_session()
        session.add_all.side_effect = lambda items: added.extend(items)

        records = [("BOS", "nba", [_player(is_key=False)])]
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._persist_injury_reports(records)

        assert added[0].status == "questionable"
        assert added[0].impact_score == 0.4

    def test_player_id_truncated_to_50_chars(self):
        updater = _updater()
        added = []
        session = _mock_session()
        session.add_all.side_effect = lambda items: added.extend(items)

        records = [("LAL", "nba", [_player(name="A Very Long Player Name That Exceeds Limits XYZ")])]
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._persist_injury_reports(records)

        assert len(added[0].player_id) <= 50

    def test_multiple_players_one_team_all_added(self):
        updater = _updater()
        added = []
        session = _mock_session()
        session.add_all.side_effect = lambda items: added.extend(items)

        players = [_player("Player A", True), _player("Player B", False)]
        records = [("LAL", "nba", players)]
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._persist_injury_reports(records)

        assert len(added) == 2

    def test_duplicate_team_deduped_first_entry_wins(self):
        """Same team appearing twice (home + away in same batch) → only persisted once."""
        updater = _updater()
        added = []
        session = _mock_session()
        session.add_all.side_effect = lambda items: added.extend(items)

        records = [
            ("LAL", "nba", [_player("Player A")]),
            ("LAL", "nba", [_player("Player B")]),  # duplicate
        ]
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._persist_injury_reports(records)

        # Only first LAL entry kept — one player
        assert len(added) == 1
        assert added[0].player_name == "Player A"

    def test_two_different_teams_both_added(self):
        updater = _updater()
        added = []
        session = _mock_session()
        session.add_all.side_effect = lambda items: added.extend(items)

        records = [
            ("LAL", "nba", [_player("Player A")]),
            ("BOS", "nba", [_player("Player B")]),
        ]
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._persist_injury_reports(records)

        assert len(added) == 2
        teams = {r.team_id for r in added}
        assert teams == {"LAL", "BOS"}

    def test_session_committed(self):
        updater = _updater()
        session = _mock_session()

        records = [("LAL", "nba", [_player()])]
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._persist_injury_reports(records)

        session.commit.assert_called_once()

    def test_db_error_swallowed(self):
        updater = _updater()
        session = _mock_session()
        session.__enter__.side_effect = RuntimeError("DB unavailable")

        records = [("LAL", "nba", [_player()])]
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._persist_injury_reports(records)  # must not raise

    def test_comment_truncated_to_100_chars(self):
        updater = _updater()
        added = []
        session = _mock_session()
        session.add_all.side_effect = lambda items: added.extend(items)

        long_comment = "X" * 200
        records = [("LAL", "nba", [_player(comment=long_comment)])]
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._persist_injury_reports(records)

        assert len(added[0].injury_type) <= 100


# ---------------------------------------------------------------------------
# _settle_predictions()
# ---------------------------------------------------------------------------

class TestSettlePredictions:

    # ------------------------------------------------------------------
    # Session mock factory
    # ------------------------------------------------------------------

    def _session_with_game(
        self,
        game_id_in_db: str | None = "0021234567",
        total_line: float | None = 218.5,
        predictions: list | None = None,
    ) -> MagicMock:
        """
        Build a mock session whose execute() chain returns:
          1st call  → select(Game)   → scalar_one_or_none() → mock Game or None
          2nd call  → update(Game)   → (return ignored)
          3rd call  → select(Prediction) → scalars().all() → list of Preds
          4th+ call → update(Prediction) → (return ignored)
        """
        session = _mock_session()

        if game_id_in_db is None:
            # No game found in DB
            game_result = MagicMock()
            game_result.scalar_one_or_none.return_value = None
            session.execute.return_value = game_result
            return session

        mock_game = MagicMock()
        mock_game.id = uuid.uuid4()
        mock_game.total_line = total_line

        game_result = MagicMock()
        game_result.scalar_one_or_none.return_value = mock_game

        pred_result = MagicMock()
        pred_result.scalars.return_value.all.return_value = predictions or []

        # Returns alternate between game select, game update, pred select, pred updates
        side_effects = [game_result, MagicMock(), pred_result] + [MagicMock()] * 10
        session.execute.side_effect = side_effects
        return session

    # ------------------------------------------------------------------
    # No-op cases
    # ------------------------------------------------------------------

    def test_no_game_in_db_returns_zero(self):
        updater = _updater()
        session = self._session_with_game(game_id_in_db=None)
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            result = updater._settle_predictions(_games_df())
        assert result == 0

    def test_game_without_total_line_skipped(self):
        updater = _updater()
        session = self._session_with_game(total_line=None)
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            result = updater._settle_predictions(_games_df())
        assert result == 0

    def test_no_pending_predictions_returns_zero(self):
        updater = _updater()
        session = self._session_with_game(predictions=[])
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            result = updater._settle_predictions(_games_df())
        assert result == 0

    # ------------------------------------------------------------------
    # Settlement logic
    # ------------------------------------------------------------------

    def test_over_prediction_correct_when_total_exceeds_line(self):
        """220 total > 218.5 line → result_over; 'over' prediction is correct."""
        updater = _updater()
        pred = MagicMock()
        pred.id = uuid.uuid4()
        pred.predicted_direction = "over"

        session = self._session_with_game(total_line=218.5, predictions=[pred])
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            result = updater._settle_predictions(_games_df(total=[220], home_score=[112], away_score=[108]))

        assert result == 1

    def test_returns_count_of_settled_predictions(self):
        """Two pending predictions for the same game → count == 2."""
        updater = _updater()
        preds = [MagicMock(id=uuid.uuid4(), predicted_direction="over") for _ in range(2)]
        for p in preds:
            p.predicted_direction = "over"

        session = self._session_with_game(total_line=218.5, predictions=preds)
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            result = updater._settle_predictions(_games_df(total=[220], home_score=[112], away_score=[108]))

        assert result == 2

    def test_under_prediction_correct_when_total_below_line(self):
        """210 total < 218.5 line → result_under; 'under' prediction is correct → settled == 1."""
        updater = _updater()
        pred = MagicMock(id=uuid.uuid4(), predicted_direction="under")

        session = self._session_with_game(total_line=218.5, predictions=[pred])
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            result = updater._settle_predictions(_games_df(total=[210], home_score=[105], away_score=[105]))

        assert result == 1

    def test_multiple_games_all_settled(self):
        """Two rows in new_games DF, each with a matching DB game and one prediction."""
        updater = _updater()

        game_a_id = uuid.uuid4()
        game_b_id = uuid.uuid4()

        def _make_game(gid, line):
            g = MagicMock()
            g.id = gid
            g.total_line = line
            return g

        pred_a = MagicMock(id=uuid.uuid4(), predicted_direction="over")
        pred_b = MagicMock(id=uuid.uuid4(), predicted_direction="under")

        session = _mock_session()
        # Sequence: game_A select, game_A update, pred_A select, pred_A update,
        #            game_B select, game_B update, pred_B select, pred_B update
        def _make_game_result(g):
            r = MagicMock()
            r.scalar_one_or_none.return_value = g
            return r

        def _make_pred_result(preds):
            r = MagicMock()
            r.scalars.return_value.all.return_value = preds
            return r

        session.execute.side_effect = [
            _make_game_result(_make_game(game_a_id, 220.0)),
            MagicMock(),
            _make_pred_result([pred_a]),
            MagicMock(),
            _make_game_result(_make_game(game_b_id, 215.0)),
            MagicMock(),
            _make_pred_result([pred_b]),
            MagicMock(),
        ]

        df = pd.DataFrame({
            "game_id": ["GA001", "GB002"],
            "home_score": [113, 102],
            "away_score": [109, 108],
            "total": [222, 210],
        })
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            result = updater._settle_predictions(df)

        assert result == 2
        session.commit.assert_called_once()

    def test_session_committed_after_all_rows(self):
        updater = _updater()
        pred = MagicMock(id=uuid.uuid4(), predicted_direction="over")
        session = self._session_with_game(total_line=218.5, predictions=[pred])

        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            updater._settle_predictions(_games_df())

        session.commit.assert_called_once()

    def test_db_error_swallowed_returns_zero(self):
        updater = _updater()
        session = _mock_session()
        session.__enter__.side_effect = RuntimeError("connection refused")

        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            result = updater._settle_predictions(_games_df())

        assert result == 0

    def test_total_derived_from_scores_when_total_col_missing(self):
        """When 'total' column absent, falls back to home_score + away_score."""
        updater = _updater()
        pred = MagicMock(id=uuid.uuid4(), predicted_direction="over")
        # line=218.5; home 115 + away 108 = 223 → over → settled
        session = self._session_with_game(total_line=218.5, predictions=[pred])

        df = pd.DataFrame({"game_id": ["0021234567"], "home_score": [115], "away_score": [108]})
        with patch("proedge.db.session.SyncSessionLocal", return_value=session):
            result = updater._settle_predictions(df)

        assert result == 1
