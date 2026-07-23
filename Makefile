UV_RUN ?= uv run

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
	$(UV_RUN) ruff check phy

format:
	$(UV_RUN) ruff format phy

format-check:
	$(UV_RUN) ruff format --check phy

lint-fix:
	$(UV_RUN) ruff check --fix phy

# Test everything except apps
test: lint format-check
	$(UV_RUN) pytest -xvv --cov-report= --cov=phy phy --ignore=phy/apps/ --cov-append
	$(UV_RUN) coverage report --omit "*/phy/apps/*,*/phy/plot/gloo/*"

# Test just the apps
test-apps: lint
	$(UV_RUN) pytest --cov-report term-missing phy/apps/ --cov-append

# Test everything
test-full: test test-apps
	$(UV_RUN) coverage report --omit "*/phy/plot/gloo/*"

test-fast:
	$(UV_RUN) pytest phy

doc:
	$(UV_RUN) python tools/api.py && $(UV_RUN) python tools/extract_shortcuts.py && $(UV_RUN) python tools/plugins_doc.py

doc-check: doc
	$(UV_RUN) python tools/check_docs.py
	$(UV_RUN) mkdocs build --strict
	git diff --exit-code -- docs/ plugins/README.md

build:
	uv build

upload:
	uv publish

upload-test:
	uv publish --publish-url https://test.pypi.org/legacy/

publish-test:
	python3 scripts/release_publish.py publish-testpypi-dev

version-test:
	python3 scripts/release_publish.py print-latest-testpypi-version

publish-pypi:
	python3 scripts/release_publish.py publish-pypi

coverage:
	$(UV_RUN) coverage html

dev: install lint format test

ci: lint format-check test-full build

RELEASE_SMOKE_DATASET ?= $(CURDIR)/../phy-data/template
PYPROJECT_VERSION := $(shell python3 scripts/release_publish.py print-current-version)
SMOKE_VERSION ?=
PYPI_SMOKE_VERSION = $(or $(SMOKE_VERSION),$(PYPROJECT_VERSION))
TEST_SMOKE_VERSION = $(or $(SMOKE_VERSION),$(shell python3 scripts/release_publish.py print-latest-testpypi-version))
RELEASE_SMOKE_INDEX_URL ?=
RELEASE_SMOKE_EXTRA_INDEX_URL ?=

define run_smoke
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(1)" \
	RELEASE_SMOKE_VERSION="$(2)" \
	RELEASE_SMOKE_INDEX_URL="$(3)" \
	RELEASE_SMOKE_EXTRA_INDEX_URL="$(4)" \
	bash scripts/release_smoke_test.sh $(5)
endef

define run_open
	RELEASE_SMOKE_DATASET="$(RELEASE_SMOKE_DATASET)" \
	RELEASE_SMOKE_ENV="$(1)" \
	bash scripts/release_smoke_test.sh open
endef

smoke-local: build
	$(call run_smoke,$(CURDIR)/.release-smoke/local,,,,local)

open-local:
	$(call run_open,$(CURDIR)/.release-smoke/local)

smoke-pypi:
	$(call run_smoke,$(CURDIR)/.release-smoke/pypi-$(PYPI_SMOKE_VERSION),$(PYPI_SMOKE_VERSION),$(RELEASE_SMOKE_INDEX_URL),$(RELEASE_SMOKE_EXTRA_INDEX_URL),pypi)

open-pypi:
	$(call run_open,$(CURDIR)/.release-smoke/pypi-$(PYPI_SMOKE_VERSION))

smoke-test:
	$(call run_smoke,$(CURDIR)/.release-smoke/testpypi-$(TEST_SMOKE_VERSION),$(TEST_SMOKE_VERSION),https://test.pypi.org/simple/,https://pypi.org/simple/,pypi)

open-test:
	$(call run_open,$(CURDIR)/.release-smoke/testpypi-$(TEST_SMOKE_VERSION))

.PHONY: clean-build clean-pyc clean-test clean install lint format format-check lint-fix test test-apps test-full test-fast doc doc-check build upload upload-test publish-test version-test publish-pypi coverage dev ci smoke-local open-local smoke-pypi open-pypi smoke-test open-test
