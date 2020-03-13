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

# Test everything except apps.
test: lint
	py.test -xvv --cov-report= --cov=phy phy --ignore=phy/apps/ --cov-append
	coverage report --omit */phy/apps/*,*/phy/plot/gloo/*

# Test just the apps.
test-apps: lint
	py.test --cov-report term-missing --cov=phy.apps phy/apps/ --cov-append

# Test everything.
test-full: test test-apps
	coverage report --omit */phy/plot/gloo/*

doc:
	python tools/api.py && python tools/extract_shortcuts.py && python tools/plugins_doc.py

build:
	python setup.py sdist --formats=zip

upload:
	python setup.py sdist --formats=zip upload
