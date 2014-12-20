## Contribute

Please read this entire document before making any code contribution.

### Python

On your development computer:

* Use the very latest Anaconda (`conda update conda anaconda`).
* Use a special `phy` conda environment based on the latest Python 3.x (3.4 at the time of writing).
* Have another `phy2` clone environment based on Python 2.7.
* phy only supports Python 2.7, Python 3.3, Python 3.4.
* Use `phy.ext.six` for writing compatible code (see [the documentation here](http://pythonhosted.org/six/))
* You need the following dependencies for development (not required for using phy): nose, pip, flake8, coverage, coveralls.
* For IPython, only use IPython 3.0 (or master until 3.0 is released in early 2015).

A few rules:

* Every module `phy/mypackage/mymodule.py` must come with a `phy/mypackage/tests/test_mymodule.py` test module that contains a bunch of `test_*()` functions.
* Never import test modules in the main code.
* Do not import packages from `phy/__init__.py`. Every subpackage `phy.stuff` will need to be imported explicitely by the user. Dependencies required by this subpackage will only be loaded when the subpackage is loaded. This ensures that users can use `phy.subpackageA` without having to install the dependencies required by `phy.subpackageB`.
* phy's required dependencies are: numpy. Every subpackage can come with further dependencies. For example, `phy.io` depends on h5py.
* You can experiment with ideas and prototypes in the `kwikteam/experimental` repo. Use a different folder for every experiment.
* `kwikteam/phy` will only contain a `master` branch and release tags. There should be no experimental/debugging code in the entire repository.


### GitHub flow

* Work through PRs from `yourfork/specialbranch` against `phy/master` exclusively.
* Set `upstream` to `kwikteam/phy` and `origin` to your fork.
* When master and your PR's branch are out of sync, [rebase your branch in your fork](https://groups.google.com/forum/#!msg/vispy-dev/q-UNjxburGA/wYNkZRXiySwJ).
* Two-pairs-of-eyes rules: every line of code needs to be reviewed by 2 people, including the author.
* Never merge your own PR (but exceptions happen).
* Many small commits are better than few huge commits.
* A PR is assumed to be **not ready for merge** unless explicitely stated otherwise.
* Always run `make test` before stating that a PR is ready to merge (and ideally before pushing on your PR's branch).
* We try to have a code coverage close to 100%: always test all features you implement, and verify through code coverage that all lines are covered by your tests.
* Always wait for Travis to be green before merging.
* `phy/master` should always be stable and deployable.
* Use semantic versioning for stable releases.
* Do not make too many releases until the software is mature enough. Early adopters can work directly on master.
* We shall have a system that automatically registers the commit hash used to analyze any dataset.
* We follow almost all [PEP8 rules](https://www.python.org/dev/peps/pep-0008/), except [for a few exceptions](https://github.com/kwikteam/phy/blob/master/Makefile#L24).


### Text editor

Make sure your text editor is configured to:

* automatically clean blank lines (containing only spaces)
* only use spaces and **never** tab indents
* enforce an empty blank line at the end of every text file
* display a vertical ruler at 79 characters (size limit of every line)
