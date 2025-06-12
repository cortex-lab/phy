clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	rm -fr .eggs/

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache/
	rm -fr .ruff_cache/

clean: clean-build clean-pyc clean-test

install:
	uv sync --dev --extra qt5

install-qt6:
	uv sync --dev --extra qt6

lint:
	uv run ruff check phy

format:
	uv run ruff format phy

format-check:
	uv run ruff format --check phy

lint-fix:
	uv run ruff check --fix phy

# Test everything except apps
test: lint format-check
	uv run pytest -xvv --cov-report= --cov=phy phy --ignore=phy/apps/ --cov-append
	uv run coverage report --omit "*/phy/apps/*,*/phy/plot/gloo/*"

# Test just the apps
test-apps: lint
	uv run pytest --cov-report term-missing --cov=phy.apps phy/apps/ --cov-append

# Test everything
test-full: test test-apps
	uv run coverage report --omit "*/phy/plot/gloo/*"

test-fast:
	uv run pytest phy

doc:
	uv run python tools/api.py && uv run python tools/extract_shortcuts.py && uv run python tools/plugins_doc.py

build:
	uv build

upload:
	uv publish

upload-test:
	uv publish --publish-url https://test.pypi.org/legacy/

coverage:
	uv run coverage html

dev: install lint format test

ci: lint format-check test-full build

.PHONY: clean-build clean-pyc clean-test clean install install-qt6 lint format format-check lint-fix test test-apps test-full test-fast doc build upload upload-test coverage dev ci