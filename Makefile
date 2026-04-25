.PHONY: install dev up down migrate train test lint

install:
	pip install -e ".[dev]"

dev:
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

test:
	pytest tests/ -v --cov=src/proedge --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
