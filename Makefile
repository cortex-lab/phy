ifeq ($(OS),Windows_NT)
SHELL := C:/Program Files/Git/bin/bash.exe
.SHELLFLAGS := -lc
PYTHON_BIN ?= python
else
PYTHON_BIN ?= python3
endif

export PYTHON ?= $(PYTHON_BIN)

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

publish-test:
	$(PYTHON_BIN) scripts/release_publish.py publish-testpypi-dev

version-test:
	$(PYTHON_BIN) scripts/release_publish.py print-latest-testpypi-version

publish-pypi:
	$(PYTHON_BIN) scripts/release_publish.py publish-pypi

coverage:
	uv run coverage html

dev: install lint format test

ci: lint format-check test-full build

RELEASE_SMOKE_DATASET ?= $(CURDIR)/../phy-data/template
PYPROJECT_VERSION := $(shell $(PYTHON_BIN) scripts/release_publish.py print-current-version)
SMOKE_VERSION ?=
PYPI_SMOKE_VERSION = $(or $(SMOKE_VERSION),$(PYPROJECT_VERSION))
TEST_SMOKE_VERSION = $(or $(SMOKE_VERSION),$(shell $(PYTHON_BIN) scripts/release_publish.py print-latest-testpypi-version))
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

.PHONY: clean-build clean-pyc clean-test clean install lint format format-check lint-fix test test-apps test-full test-fast doc build upload upload-test publish-test version-test publish-pypi coverage dev ci smoke-local open-local smoke-pypi open-pypi smoke-test open-test
