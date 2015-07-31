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
	flake8 phy

test: lint
	python setup.py test

coverage:
	coverage --html

test-quick: lint
	python setup.py test -a "-m \"not long\" phy"

unit-tests: lint
	python setup.py test -a phy

integration-tests: lint
	python setup.py test -a tests

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
