"""Restore private demo model artifacts from Neon at container startup."""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import psycopg2


def main() -> None:
    url = os.environ["DATABASE_URL_SYNC"]
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
