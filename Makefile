FLAKE8 ?= flake8

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "release - package and upload a release"
	@echo "apidoc - build API doc"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

lint:
	${FLAKE8} phy --exclude=phy/ext/* --ignore=E226,E265,F811

test: lint
	py.test --cov-report term-missing --cov phy -s

coverage:
	coverage --html

test-quick: lint
	py.test phy -m "not long"

apidoc:
	python tools/api.py

build:
	python setup.py sdist --formats=zip

upload:
	python setup.py sdist --formats=zip upload

release-test:
	python tools/release.py release_test

release:
	python tools/release.py release
