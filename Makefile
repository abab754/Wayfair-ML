.PHONY: setup test lint serve docker data eval

setup:
	uv venv .venv
	uv pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/

data:
	python -m wands.data.job

eval:
	@echo "TODO: run eval harness (MODEL=$(MODEL))"

serve:
	@echo "TODO: run FastAPI locally"

docker:
	@echo "TODO: build + run container"
