"""API key authentication middleware.

If settings.api_key is set, requests to /predictions, /training, and
/backtest must include an X-API-Key header matching the configured key.
Requests to /health, /metrics, /docs, /openapi.json, /lines, and
/performance are public.
"""
from __future__ import annotations

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_PROTECTED_PREFIXES = ("/predictions", "/training", "/backtest")
_PUBLIC_PREFIXES = ("/health", "/metrics", "/docs", "/openapi.json", "/redoc", "/")


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if not self._api_key:
            return await call_next(request)

        path = request.url.path
        protected = any(path.startswith(p) for p in _PROTECTED_PREFIXES)
        if not protected:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != self._api_key:
            return Response(
                content='{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return await call_next(request)
