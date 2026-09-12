"""Restore private demo model artifacts from Neon at container startup."""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2


def normalize_neon_url(url: str) -> str:
    """Repair a Render value truncated immediately before Neon's fixed DNS suffix."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if not host.endswith(".us-east-2"):
        return url
    auth = parts.username or ""
    if parts.password:
        auth += f":{parts.password}"
    if auth:
        auth += "@"
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{auth}{host}.aws.neon.tech{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def main() -> None:
    url = normalize_neon_url(os.environ["DATABASE_URL_SYNC"])
    with psycopg2.connect(url) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT payload FROM demo_model_artifacts WHERE name = %s", ("proedge-v1",))
        row = cursor.fetchone()
    if not row:
        raise RuntimeError("Private ProEdge model artifact is unavailable")
    root = Path("models").resolve()
    root.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(bytes(row[0])), mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError("Unsafe model artifact path")
        archive.extractall(root, filter="data")


if __name__ == "__main__":
    main()
