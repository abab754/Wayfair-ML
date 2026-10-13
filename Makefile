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
	python -c "from wands.retrieval.bm25 import BM25Retriever; from wands.eval.runner import evaluate, save_results; r = BM25Retriever(); res = evaluate(r); res['model']='bm25'; save_results(res, 'sprint1_bm25')"

serve:
	@echo "TODO: run FastAPI locally"

docker:
	@echo "TODO: build + run container"
