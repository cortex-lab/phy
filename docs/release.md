# Release notes

The current stable release is phy 2.1.0, published on 17 July 2026. The source
tree is currently developing phy 2.1.1.

This page describes the latest stable release. For a version-by-version list
of user-visible changes, including work not yet released, see the
[project changelog](changelog.md).
Maintainer publishing instructions are kept separately in the
[release workflow](releasing.md).

## phy 2.1.0

Phy 2.1.0 is a maintenance release that makes phy easier to install and more
reliable on current systems. It incorporates fixes identified during the
2.1.0 release-candidate testing period.

### Highlights

- Modernized dependencies, packaging, and continuous integration for Python
  3.10 and newer.
- Replaced the fragile legacy web-based Cluster View with a native Qt
  implementation.
- Improved GUI startup, rendering, and resource cleanup across Linux, macOS,
  and Windows.
- Kept dataset and file formats unchanged.

### Added

- A Code of Conduct for project participation and incident reporting.

### Fixed

- Preserve the Cluster View sort order when cluster metadata changes
  ([#1375](https://github.com/cortex-lab/phy/issues/1375)).
- Display NumPy-typed values correctly in native Qt Cluster View columns
  ([#1377](https://github.com/cortex-lab/phy/issues/1377)).

### Install or upgrade

Use a fresh Python 3.10 or newer environment when possible, then install the
exact stable release:

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade "phy==2.1.0"
phy --version
```

The release is available from
[PyPI](https://pypi.org/project/phy/2.1.0/). Tagged source, release notes, and
verified distribution files are available from the
[GitHub release](https://github.com/cortex-lab/phy/releases/tag/v2.1.0).

### Compatibility

Dataset formats are unchanged. Plugins using supported Python-side controller,
event, or view APIs are expected to continue working. Plugins that depend on
internal HTML or other components of the legacy web-based GUI may require
updates.

Please report regressions on the
[GitHub issue tracker](https://github.com/cortex-lab/phy/issues). Include the
operating system, Python version, installation method, whether plugins are
enabled, and a minimal error or reproduction.

## Historical releases

The [changelog](changelog.md) contains concise summaries of every tagged
release. Detailed commit-level history and artifacts remain available on
[GitHub](https://github.com/cortex-lab/phy/releases).
