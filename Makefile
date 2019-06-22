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
	py.test --cov-report term-missing --cov=phy phy

doc:
	python tools/api.py && python tools/extract_shortcuts.py && python tools/plugins_doc.py

build:
	python setup.py sdist --formats=zip

upload:
	python setup.py sdist --formats=zip upload
