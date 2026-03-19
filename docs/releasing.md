# Maintainer release workflow

This page is for maintainers preparing TestPyPI or PyPI releases. It is not part of the
user-facing release notes.

## Local release smoke test

The repository contains a local packaging smoke test that mirrors the end-user install flow in an
isolated virtual environment and uses the small template dataset from `../phy-data/template/`.

Run this before upload to validate the built wheel:

```bash
make release-smoke-local
make release-open-local
```

`make release-smoke-local` builds the wheel, creates `.release-smoke/local`, installs `phy` from
the local wheel into that fresh environment, checks imports and CLI entry points, and runs:

```bash
phy template-describe ../phy-data/template/params.py
```

`make release-open-local` launches the GUI from that isolated environment so that you can confirm
the dataset opens correctly.

After upload to TestPyPI or PyPI, validate the published package in another fresh environment:

```bash
make release-smoke-pypi
make release-open-pypi
```

For TestPyPI, use the convenience targets:

```bash
make release-smoke-testpypi
make release-open-testpypi
```

By default, these commands read the current version from `pyproject.toml`. You can still override
`RELEASE_SMOKE_VERSION` manually if you need to test a different published version.

This validates the intended user path: a plain install in a fresh environment, followed by opening
a dataset with `phy template-gui`.

## Disposable TestPyPI dev releases

TestPyPI does not let you overwrite an existing file for the same version, so the repository
includes a disposable dev-release workflow around the current checked-in release candidate version.

Before publishing, provide a TestPyPI API token. Username/password uploads are rejected by
TestPyPI. The helper accepts `TESTPYPI_TOKEN`, `TEST_PYPI_TOKEN`, or `UV_PUBLISH_TOKEN`, and also
falls back to `~/.pypirc` under `[testpypi]` when it contains:

```bash
username = __token__
password = pypi-...
```

Run:

```bash
make release-publish-testpypi-dev
```

This command:

* reads the current version from `pyproject.toml` (for example `2.1.0rc1`)
* queries TestPyPI for existing `2.1.0rc1.devN` releases
* picks the next free version such as `2.1.0rc1.dev3`
* creates a temporary staged copy of the repository
* updates the version in that temporary copy only
* builds and publishes that disposable version to TestPyPI
* records the published version in `.release-smoke/latest-testpypi-version.txt`

Your working tree keeps the original final candidate version unchanged.

After publishing the disposable TestPyPI build, validate exactly that uploaded version with:

```bash
make release-smoke-testpypi-latest
make release-open-testpypi-latest
```

You can print the recorded version directly with:

```bash
make release-latest-testpypi-version
```

## Final PyPI publish

Once the disposable TestPyPI builds have been validated on your different operating systems and you
are ready to publish the exact checked-in version from `pyproject.toml`, run:

The final publish helper accepts `PYPI_TOKEN` or `UV_PUBLISH_TOKEN`, and also falls back to
`~/.pypirc` under `[pypi]` with:

```bash
username = __token__
password = pypi-...
```

Then run:

```bash
make release-publish-pypi
```

This target refuses to publish if the checked-in version still contains `.dev`.

## Typical RC release checklist

For a normal release-candidate cycle, the usual command sequence is:

```bash
# 1. Local code and packaging checks
make test-fast
make build
make release-smoke-local

# 2. Publish a disposable TestPyPI build for this RC line
make release-publish-testpypi-dev

# 3. Verify that uploaded TestPyPI build on this machine
make release-smoke-testpypi-latest
make release-open-testpypi-latest

# 4. Repeat step 3 on your other OS machines
make release-latest-testpypi-version
# then on the other machine, after recording that version locally:
make release-smoke-testpypi-latest
make release-open-testpypi-latest

# 5. Once everything is green, publish the checked-in RC version to real PyPI
make release-publish-pypi

# 6. Verify the real PyPI release
make release-smoke-pypi
make release-open-pypi
```

If a TestPyPI upload fails validation, fix the issue locally and run
`make release-publish-testpypi-dev` again. It will automatically choose the next free `.devN`
version without changing the checked-in RC version.

If another machine does not yet have `.release-smoke/latest-testpypi-version.txt`, you can still
fall back to:

```bash
make release-smoke-testpypi RELEASE_SMOKE_VERSION=<version>
make release-open-testpypi RELEASE_SMOKE_VERSION=<version>
```
