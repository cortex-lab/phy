clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean: clean-build clean-pyc

lint:
	flake8 phy

test: lint
	python setup.py test

coverage:
	coverage --html

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
