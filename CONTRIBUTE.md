## Contribute

Please read this entire document before contributing any code.

### Python

On your development computer:

* Use the very latest Anaconda release (`conda update conda anaconda`).
* Use a special `phy` conda environment based on Python 3.5.
* Have another `phy2` clone environment based on Python 2.7.
* Install the dev dependencies: `pip install -r requirements-dev.txt`
* phy only supports Python 2.7 and Python 3.5+.
* Use the `six` library for writing compatible code (see [the documentation here](http://pythonhosted.org/six/))

A few rules:

* Every module `phy/mypackage/mymodule.py` must come with a `phy/mypackage/tests/test_mymodule.py` test module that contains a bunch of `test_*()` functions.
* Never import test modules in the main code.
* At least **99% code coverage required**. Always test all features you implement, and verify through code coverage that all lines are covered by your tests. Use `#pragma: no cover` comments for lines that don't absolutely need to be covered (for example, rare exception-raising code).
* In general, do not import packages from `phy/__init__.py`. Every subpackage `phy.stuff` will need to be imported explicitly by the user. Dependencies required by this subpackage will only be loaded when the subpackage is loaded. This ensures that users can use `phy.subpackageA` without having to install the dependencies required by `phy.subpackageB`.
* phy's required dependencies are: pip, traitlets, click, numpy. Every subpackage may come with further dependencies.
* `kwikteam/phy` will only contain a `master` branch, release tags, and possibly one refactoring branch.


### GitHub flow

* Work through PRs from `yourfork/specialbranch` against `phy/master`.
* Set `upstream` to `kwikteam/phy` and `origin` to your fork.
* When master and your PR's branch are out of sync, [rebase your branch in your fork](https://groups.google.com/forum/#!msg/vispy-dev/q-UNjxburGA/wYNkZRXiySwJ).
* Two-pairs-of-eyes rule: every line of code needs to be reviewed by 2 people, including the author.
* Never merge your own PR to the main phy repository, unless in exceptional circumstances.
* A PR is assumed to be **not ready for merge** unless explicitly stated otherwise.
* Always run `make test` before stating that a PR is ready to merge (and ideally before pushing on your PR's branch).
* Always wait for Travis to be green before merging.
* `phy/master` should always be stable and deployable.
* Use [semantic versioning](http://www.semver.org) for stable releases.
* We follow almost all [PEP8 rules](https://www.python.org/dev/peps/pep-0008/), except [for a few exceptions](https://github.com/kwikteam/phy/blob/master/Makefile#L24).


### Git commit messages

* Many small commits are better than few huge commits: there must be exactly **one logical change per commit**. Do not mix up whitespace changes with functional code changes; do not put two unrelated changes, or different functional changes, into a single commit. It is easy to revert and cherry-pick individual commits, but splitting up commits post-hoc is much harder.
* Follow the "50/72 rule". This means your commit messages must be in the following format:
    * First line: a brief summary of the change, no more than 50 characters (the "subject line"). Example: `Added support for sparse arrays in loader`
    * If the first line does not contain all the information needed, the second line **must be blank**.
    * Third (and following) lines: a more descriptive "body" of the commit. No more than 72 characters per line.
* More details can be found [here](https://wiki.openstack.org/wiki/GitCommitMessages).


### Text editor

Make sure your text editor is configured to:

* Automatically clean blank lines (i.e. no lines containing only whitespace)
* Use four spaces per indent level, and **never** tab indents
* Enforce an empty blank line at the end of every text file
* Display a vertical ruler at 79 characters (length limit of every line)

Below is a settings script for the popular text editor [Sublime](http://www.sublimetext.com) which you can put into your ```Preferences -> Settings (User)```:

```
{
    "auto_match_enabled": false,
    "detect_indentation": false,
    "ensure_newline_at_eof_on_save": true,
    "font_size": 14,
    "indent_to_bracket": true,
    "rulers":
    [
        79,
        99,
        119
    ],
    "tab_size": 4,
    "translate_tabs_to_spaces": true,
    "trim_trailing_white_space_on_save": true,
    "word_wrap": true,
    "wrap_width": 80
}
```
