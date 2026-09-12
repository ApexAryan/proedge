FROM python:3.11-slim

# LightGBM requires OpenMP; XGBoost requires libgomp on linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY scripts/load_demo_models.py scripts/load_demo_models.py
RUN pip install --no-cache-dir -e .

COPY alembic/ alembic/
COPY alembic.ini .

ENV PYTHONPATH=/app/src

CMD ["sh", "-c", "python scripts/load_demo_models.py && alembic upgrade head && uvicorn proedge.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
