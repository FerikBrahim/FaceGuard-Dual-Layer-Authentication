install:
	python -m pip install -e .

install-dev:
	python -m pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check src scripts tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
