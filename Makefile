.PHONY: help build up down logs clean dev dev-down test health

# Default target
help:
	@echo "Platinum-Palladium AI Printing Tool - Docker Commands"
	@echo ""
	@echo "Production:"
	@echo "  make build       Build production Docker image"
	@echo "  make up          Start API service (production)"
	@echo "  make down        Stop all services"
	@echo "  make logs        View API logs"
	@echo "  make health      Check API health status"
	@echo ""
	@echo "Development:"
	@echo "  make dev         Start API + frontend dev server"
	@echo "  make dev-down    Stop dev services"
	@echo "  make shell       Open shell in API container"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       Remove containers, volumes, and images"
	@echo "  make rebuild     Clean rebuild of all images"
	@echo "  make test        Run tests inside container"
	@echo "  make backup      Backup data volume"
	@echo ""

# Production targets
build:
	docker compose build

up:
	docker compose up -d api
	@echo "API started at http://localhost:8000"
	@echo "Health check: curl http://localhost:8000/api/health"

down:
	docker compose down

logs:
	docker compose logs -f api

health:
	@docker compose ps | grep api || echo "Container not running"
	@curl -s http://localhost:8000/api/health | jq . || echo "API not responding"

# Development targets
dev:
	docker compose --profile dev up -d
	@echo "API:      http://localhost:8000"
	@echo "Frontend: http://localhost:5173"

dev-down:
	docker compose --profile dev down

shell:
	docker compose exec api /bin/bash

# Testing
test:
	docker compose exec api pytest tests/ -v

# Maintenance targets
clean:
	docker compose down -v
	docker compose rm -f
	docker volume prune -f

rebuild: clean
	docker compose build --no-cache

backup:
	@mkdir -p backups
	docker compose exec api tar czf - /app/data > backups/ptpd-data-$$(date +%Y%m%d-%H%M%S).tar.gz
	@echo "Backup created in backups/"

restore:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make restore FILE=backups/ptpd-data-YYYYMMDD-HHMMSS.tar.gz"; \
		exit 1; \
	fi
	docker compose exec -T api tar xzf - -C /app < $(FILE)
	@echo "Data restored from $(FILE)"

# Utility targets
ps:
	docker compose ps

stats:
	docker stats $$(docker compose ps -q)

prune:
	docker system prune -a --volumes -f
