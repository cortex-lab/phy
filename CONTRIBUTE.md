* Contributions with GitHub flow exclusively
    * fork the repo (even if you're a core developer)
    * create a branch in your fork with a descriptive name
    * commit locally on this branch
    * push this branch on your fork on GitHub
    * open a Pull Request against the origin repo, master branch
* origin master should always be deployable/ready to use/working
* PEP8 standards, check with `flake8` or `make lint`
* `make test` locally before stating your PR as ready for review/merge (this also checks flake8 automatically)
* add unit tests for all features, using `nose`
* tests are also automatically run in the cloud with Travis CI
* two-pair-of-eyes rule for merging PRs
* a developer cannot merge his own PR (or exceptionally)
