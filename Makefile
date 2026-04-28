.PHONY: test lint check quick-test

quick-test:
	python -m pytest tests/test_cli.py tests/test_config.py tests/test_cache.py tests/test_manifest.py -q

test:
	python -m pytest -q

lint:
	python -m ruff check src tests

check: quick-test test lint
