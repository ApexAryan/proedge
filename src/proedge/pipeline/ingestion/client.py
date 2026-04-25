import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SportsDataClient:
    """Base async HTTP client for sports data APIs with retry logic."""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()

    async def get(self, path: str, params: dict | None = None, retries: int = 3) -> Any:
        for attempt in range(retries):
            try:
                resp = await self._client.get(path, params=params or {})
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except httpx.RequestError as e:
                if attempt == retries - 1:
                    raise
                logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, retries, e)
                await asyncio.sleep(1)
        raise RuntimeError(f"Failed after {retries} retries: {path}")
