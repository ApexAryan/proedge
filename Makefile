.PHONY: install dev up down migrate train test lint env-setup update update-all web web-build start-all status

install:
	pip install -e ".[dev]"

env-setup:
	@[ -f .env ] && echo ".env already exists — skipping" || (cp .env.example .env && echo "Created .env from .env.example — fill in your keys")

dev:
	@[ -f .env ] || (echo "No .env found. Run: make env-setup" && exit 1)
	@port=$$(grep -E '^API_PORT=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r'); \
	if [ -z "$$port" ]; then port=8010; fi; \
	if lsof -ti :$$port >/dev/null 2>&1; then \
		echo "ProEdge already running on :$$port → http://localhost:$$port/dashboard"; \
		exit 0; \
	fi; \
	echo "ProEdge → http://localhost:$$port/dashboard"; \
	.venv/bin/uvicorn proedge.api.main:app --host 0.0.0.0 --port $$port --reload

start-all:
	@bash scripts/start-local-apps.sh

status:
	@kalshi=8000; \
	if [ -f "$$HOME/Desktop/kalshiedge/files/.env" ]; then \
	  k=$$(grep -E '^API_PORT=' "$$HOME/Desktop/kalshiedge/files/.env" | cut -d= -f2- | tr -d '\r'); \
	  [ -n "$$k" ] && kalshi=$$k; \
	fi; \
	proedge=8010; \
	if [ -f .env ]; then \
	  p=$$(grep -E '^API_PORT=' .env | cut -d= -f2- | tr -d '\r'); \
	  [ -n "$$p" ] && proedge=$$p; \
	fi; \
	for pair in "KalshiEdge:$$kalshi" "ProEdge:$$proedge" "EDGE NBA:8600"; do \
	  name=$${pair%%:*}; port=$${pair##*:}; \
	  if lsof -ti :$$port >/dev/null 2>&1; then echo "$$name :$$port  UP"; else echo "$$name :$$port  —"; fi; \
	done

web:
	cd web && npm install && npm run dev

web-build:
	cd web && npm install && npm run build

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
	curl -s -X POST "http://localhost:8010/training/update/$(sport)" | python3 -m json.tool

update-all:
	curl -s -X POST "http://localhost:8010/training/update/nfl" | python3 -m json.tool
	curl -s -X POST "http://localhost:8010/training/update/nba" | python3 -m json.tool
	curl -s -X POST "http://localhost:8010/training/update/mlb" | python3 -m json.tool

test:
	pytest tests/ -v --cov=src/proedge --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
