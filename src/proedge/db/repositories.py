from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from proedge.db.models import Game, InjuryReport, ModelRun, PlayerStat, Prediction


class GameRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Game:
        game = Game(**kwargs)
        self.session.add(game)
        await self.session.flush()
        return game

    async def get_by_id(self, game_id: UUID) -> Game | None:
        result = await self.session.execute(select(Game).where(Game.id == game_id))
        return result.scalar_one_or_none()

    async def get_by_external_id(self, external_id: str) -> Game | None:
        result = await self.session.execute(
            select(Game).where(Game.external_id == external_id)
        )
        return result.scalar_one_or_none()

    async def list_by_sport_date(
        self, sport: str, start: datetime, end: datetime
    ) -> list[Game]:
        result = await self.session.execute(
            select(Game)
            .where(Game.sport == sport, Game.game_date >= start, Game.game_date <= end)
            .order_by(Game.game_date)
        )
        return list(result.scalars().all())

    async def get_unresolved(self, sport: str) -> list[Game]:
        result = await self.session.execute(
            select(Game).where(
                Game.sport == sport,
                Game.status == "final",
                Game.result_over.is_(None),
                Game.total_line.isnot(None),
            )
        )
        return list(result.scalars().all())


class PredictionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> Prediction:
        pred = Prediction(**kwargs)
        self.session.add(pred)
        await self.session.flush()
        return pred

    async def get_by_game(self, game_id: UUID) -> list[Prediction]:
        result = await self.session.execute(
            select(Prediction)
            .where(Prediction.game_id == game_id)
            .order_by(Prediction.predicted_at.desc())
        )
        return list(result.scalars().all())

    async def mark_correct(self, prediction_id: UUID, is_correct: bool) -> None:
        await self.session.execute(
            update(Prediction)
            .where(Prediction.id == prediction_id)
            .values(is_correct=is_correct)
        )

    async def accuracy_by_version(self, model_version: str, sport: str) -> dict:
        result = await self.session.execute(
            select(Prediction).where(
                Prediction.model_version == model_version,
                Prediction.sport == sport,
                Prediction.is_correct.isnot(None),
            )
        )
        preds = result.scalars().all()
        if not preds:
            return {"accuracy": None, "total": 0}
        correct = sum(1 for p in preds if p.is_correct)
        return {"accuracy": correct / len(preds), "total": len(preds)}


class ModelRunRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> ModelRun:
        run = ModelRun(**kwargs)
        self.session.add(run)
        await self.session.flush()
        return run

    async def get_active(self, sport: str) -> ModelRun | None:
        result = await self.session.execute(
            select(ModelRun).where(
                ModelRun.sport == sport, ModelRun.is_active == True
            )
        )
        return result.scalar_one_or_none()

    async def deactivate_all(self, sport: str) -> None:
        await self.session.execute(
            update(ModelRun)
            .where(ModelRun.sport == sport)
            .values(is_active=False)
        )

    async def activate(self, version: str) -> None:
        await self.session.execute(
            update(ModelRun).where(ModelRun.version == version).values(is_active=True)
        )

    async def list_by_sport(self, sport: str) -> list[ModelRun]:
        result = await self.session.execute(
            select(ModelRun).where(ModelRun.sport == sport).order_by(ModelRun.trained_at.desc())
        )
        return list(result.scalars().all())


class PlayerStatRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_upsert(self, stats: list[dict]) -> None:
        for s in stats:
            self.session.add(PlayerStat(**s))
        await self.session.flush()

    async def get_recent_by_player(
        self, player_id: str, sport: str, limit: int = 20
    ) -> list[PlayerStat]:
        result = await self.session.execute(
            select(PlayerStat)
            .where(PlayerStat.player_id == player_id, PlayerStat.sport == sport)
            .order_by(PlayerStat.game_date.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recent_by_team(
        self, team_id: str, sport: str, limit: int = 20
    ) -> list[PlayerStat]:
        result = await self.session.execute(
            select(PlayerStat)
            .where(PlayerStat.team_id == team_id, PlayerStat.sport == sport)
            .order_by(PlayerStat.game_date.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class InjuryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, **kwargs) -> InjuryReport:
        existing = await self.session.execute(
            select(InjuryReport).where(
                InjuryReport.player_id == kwargs["player_id"],
                InjuryReport.sport == kwargs["sport"],
            )
        )
        record = existing.scalar_one_or_none()
        if record:
            for k, v in kwargs.items():
                setattr(record, k, v)
        else:
            record = InjuryReport(**kwargs)
            self.session.add(record)
        await self.session.flush()
        return record

    async def get_active_by_team(self, team_id: str, sport: str) -> list[InjuryReport]:
        result = await self.session.execute(
            select(InjuryReport).where(
                InjuryReport.team_id == team_id,
                InjuryReport.sport == sport,
                InjuryReport.status.in_(["out", "doubtful", "questionable"]),
            )
        )
        return list(result.scalars().all())
