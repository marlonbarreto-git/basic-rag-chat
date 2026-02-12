.PHONY: install test lint format typecheck clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/basic_rag_chat/ tests/

format:
	ruff format src/basic_rag_chat/ tests/

typecheck:
	mypy src/basic_rag_chat/

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache __pycache__ dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
