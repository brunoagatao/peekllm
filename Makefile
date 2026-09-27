.DEFAULT_GOAL := help
.PHONY: help setup test lint lint-fix format format-check typecheck check clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create/update the local virtualenv and install all dependencies (incl. dev).
	uv sync

test: ## Run the test suite.
	uv run pytest training/tests -v

lint: ## Check for lint errors (ruff).
	uv run ruff check training

lint-fix: ## Check for lint errors and auto-fix what's safe to fix.
	uv run ruff check --fix training

format: ## Format code in place (ruff format).
	uv run ruff format training

format-check: ## Check formatting without modifying files.
	uv run ruff format --check training

typecheck: ## Run static type checking (mypy).
	uv run mypy

check: lint format-check typecheck test ## Run all checks: lint, format check, typecheck, tests.

clean: ## Remove the virtualenv and Python cache artifacts.
	rm -rf .venv
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
