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
	rm -fr .release-smoke/

clean: clean-build clean-pyc clean-test

install:
	uv sync --dev

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

release-publish-testpypi-dev:
	python3 scripts/release_publish.py publish-testpypi-dev

release-latest-testpypi-version:
	python3 scripts/release_publish.py print-latest-testpypi-version

release-smoke-testpypi-latest:
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(CURDIR)/.release-smoke/testpypi-$$(python3 scripts/release_publish.py print-latest-testpypi-version)" \
	RELEASE_SMOKE_VERSION="$$(python3 scripts/release_publish.py print-latest-testpypi-version)" \
	RELEASE_SMOKE_INDEX_URL="https://test.pypi.org/simple/" \
	RELEASE_SMOKE_EXTRA_INDEX_URL="https://pypi.org/simple/" \
	bash scripts/release_smoke_test.sh pypi

release-open-testpypi-latest:
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(CURDIR)/.release-smoke/testpypi-$$(python3 scripts/release_publish.py print-latest-testpypi-version)" \
	bash scripts/release_smoke_test.sh open

release-publish-pypi:
	python3 scripts/release_publish.py publish-pypi

coverage:
	uv run coverage html

dev: install lint format test

ci: lint format-check test-full build

RELEASE_SMOKE_DATASET ?= $(CURDIR)/../phy-data/template
RELEASE_SMOKE_VERSION ?= $(shell python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
RELEASE_SMOKE_INDEX_URL ?=
RELEASE_SMOKE_EXTRA_INDEX_URL ?=

release-smoke-local: build
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(CURDIR)/.release-smoke/local" \
	bash scripts/release_smoke_test.sh local

release-open-local:
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(CURDIR)/.release-smoke/local" \
	bash scripts/release_smoke_test.sh open

release-smoke-pypi:
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(CURDIR)/.release-smoke/pypi-$(RELEASE_SMOKE_VERSION)" \
	RELEASE_SMOKE_VERSION="$(RELEASE_SMOKE_VERSION)" \
	RELEASE_SMOKE_INDEX_URL="$(RELEASE_SMOKE_INDEX_URL)" \
	RELEASE_SMOKE_EXTRA_INDEX_URL="$(RELEASE_SMOKE_EXTRA_INDEX_URL)" \
	bash scripts/release_smoke_test.sh pypi

release-open-pypi:
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(CURDIR)/.release-smoke/pypi-$(RELEASE_SMOKE_VERSION)" \
	bash scripts/release_smoke_test.sh open

release-smoke-testpypi:
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(CURDIR)/.release-smoke/testpypi-$(RELEASE_SMOKE_VERSION)" \
	RELEASE_SMOKE_VERSION="$(RELEASE_SMOKE_VERSION)" \
	RELEASE_SMOKE_INDEX_URL="https://test.pypi.org/simple/" \
	RELEASE_SMOKE_EXTRA_INDEX_URL="https://pypi.org/simple/" \
	bash scripts/release_smoke_test.sh pypi

release-open-testpypi:
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(CURDIR)/.release-smoke/testpypi-$(RELEASE_SMOKE_VERSION)" \
	bash scripts/release_smoke_test.sh open

.PHONY: clean-build clean-pyc clean-test clean install lint format format-check lint-fix test test-apps test-full test-fast doc build upload upload-test release-publish-testpypi-dev release-latest-testpypi-version release-smoke-testpypi-latest release-open-testpypi-latest release-publish-pypi coverage dev ci release-smoke-local release-open-local release-smoke-pypi release-open-pypi release-smoke-testpypi release-open-testpypi
