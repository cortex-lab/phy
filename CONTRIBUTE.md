## Contribute

### Setup

On your development computer:

* Use the latest Anaconda release with the latest version of Python (3.7 at this time).
* Install the dev dependencies: `pip install -r requirements-dev.txt`


### Testing

* Every module `phy/mypackage/mymodule.py` must come with a `phy/mypackage/tests/test_mymodule.py` test module that contains a bunch of `test_*()` functions.
* Never import test modules in the main code.
* 100% coverage is required. The `#pragma: no cover` comment can be used sparingly.
* Run `py.test phy --cov phy` to run all tests and get a coverage report.
* Run `flake8 phy` to for linting.


### Git

* There is a `master` branch (stable) and a `dev` branch (development version).
* Propose pull requests from `yourfork/specialbranch` to `phy/dev`.
* Many small commits are better than few huge commits.


### Text editor

Make sure your text editor is properly configured:

* No trailing whitespaces
* Four spaces per indent level, no tab indents
* Empty blank line at the end of every text file
* Display a vertical ruler at 99 characters (length limit of every line)
