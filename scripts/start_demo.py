"""Start the restricted demo with normalized private database endpoints."""
from __future__ import annotations

import os
import subprocess

from load_demo_models import main as load_models
from load_demo_models import normalize_neon_url


def main() -> None:
    for key in ("DATABASE_URL", "DATABASE_URL_SYNC"):
        if os.environ.get(key):
            os.environ[key] = normalize_neon_url(os.environ[key])
    load_models()
    subprocess.run(["alembic", "upgrade", "head"], check=True)
    os.execvp(
        "uvicorn",
        [
            "uvicorn",
            "proedge.api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            os.environ.get("PORT", "10000"),
        ],
    )


if __name__ == "__main__":
    main()
