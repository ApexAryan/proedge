from collections.abc import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from proedge.config import get_settings

settings = get_settings()

_async_connect_args: dict = {}
if "flycast" in settings.database_url or "fly.io" in settings.database_url:
    _async_connect_args = {"ssl": False}

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
    connect_args=_async_connect_args,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Sync session for use in thread-executor contexts (trainer, daily updater)
_sync_engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
SyncSessionLocal = sessionmaker(_sync_engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
