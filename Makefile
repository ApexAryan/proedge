.PHONY: install dev up down migrate train test lint env-setup update update-all

install:
	pip install -e ".[dev]"

env-setup:
	@[ -f .env ] && echo ".env already exists — skipping" || (cp .env.example .env && echo "Created .env from .env.example — fill in your keys")

dev:
	@[ -f .env ] || (echo "No .env found. Run: make env-setup" && exit 1)
	uvicorn proedge.api.main:app --host 0.0.0.0 --port 8000 --reload

up:
	docker compose up -d

down:
	docker compose down

migrate:
	alembic upgrade head

migrate-create:
	alembic revision --autogenerate -m "$(name)"

train:
	python -m proedge.pipeline.training.trainer --sport $(sport)

train-all:
	python -m proedge.pipeline.training.trainer --sport nfl
	python -m proedge.pipeline.training.trainer --sport nba
	python -m proedge.pipeline.training.trainer --sport mlb

update:
	curl -s -X POST "http://localhost:8000/training/update/$(sport)" | python3 -m json.tool

update-all:
	curl -s -X POST "http://localhost:8000/training/update/nfl" | python3 -m json.tool
	curl -s -X POST "http://localhost:8000/training/update/nba" | python3 -m json.tool
	curl -s -X POST "http://localhost:8000/training/update/mlb" | python3 -m json.tool

test:
	pytest tests/ -v --cov=src/proedge --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
