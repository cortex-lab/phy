# phy: interactive visualization and manual spike sorting of large-scale ephys data

[![Install and Test with Pip](https://github.com/cortex-lab/phy/actions/workflows/python-test.yml/badge.svg)](https://github.com/cortex-lab/phy/actions/workflows/python-test.yml)
[![codecov.io](https://img.shields.io/codecov/c/github/cortex-lab/phy.svg)](http://codecov.io/github/cortex-lab/phy)
[![Documentation](https://img.shields.io/badge/docs-Read_the_Docs-blue.svg)](https://phy.readthedocs.io/en/latest/)
[![GitHub release](https://img.shields.io/github/release/cortex-lab/phy.svg)](https://github.com/cortex-lab/phy/releases/latest)
[![PyPI release](https://img.shields.io/pypi/v/phy.svg)](https://pypi.python.org/pypi/phy)

[**phy**](https://github.com/cortex-lab/phy) is an open-source Python library providing a graphical user interface for visualization and manual curation of large-scale electrophysiological data. It is optimized for high-density multielectrode arrays containing hundreds to thousands of recording sites, especially Neuropixels recordings.

> **Current release:** `phy 2.1.0` is a maintenance-focused release that improves installation and GUI reliability on current systems. See the [release notes](https://phy.readthedocs.io/en/latest/release/) for details and compatibility notes.

[![phy 2.1.0 screenshot](https://user-images.githubusercontent.com/1942359/74028054-c284b880-49a9-11ea-8815-1b7e727a8644.png)](https://user-images.githubusercontent.com/1942359/74028054-c284b880-49a9-11ea-8815-1b7e727a8644.png)

## Current status

As of July 2026, `phy 2.1.0` is the current stable release for the 2.x line.

The main goals of this release are:

* dependency and packaging modernization
* replacing a fragile legacy web-based GUI component with a Qt-native implementation
* improving display reliability on modern systems
* incorporating fixes identified during release-candidate testing

Dataset formats are unchanged. Some plugins that relied on internal HTML or web-based GUI components may need updates.

Please report any issues or compatibility regressions on [GitHub issues](https://github.com/cortex-lab/phy/issues).

## Supported workflows

phy currently provides three main entry points:

* **Template GUI**: the main and recommended workflow for datasets sorted with KiloSort and Spyking Circus
* **Kwik GUI**: a legacy workflow for datasets sorted with klusta and klustakwik2
* **Trace GUI**: an experimental raw-data viewer for opening continuous electrophysiology recordings directly

Current testing and maintenance work is focused on modern Linux, macOS, and Windows environments. Linux is still the best-covered platform.

## Installation

Install phy in a fresh Python 3.10+ environment:

```bash
python -m pip install --upgrade pip
pip install phy
```

This installs the GUI runtime dependencies as part of the main package. phy
itself works with numpy 1.x and numpy 2.x; only the legacy Kwik GUI is
constrained, and those constraints live in the `kwik` extra so that they apply
to nobody else.

### Installing from a git checkout

To run phy from this repository rather than from PyPI:

```bash
conda create -n phy python=3.13 -y
conda activate phy

git clone https://github.com/cortex-lab/phy.git
cd phy
pip install -e .
```

`-e` (editable) means `git pull` updates your install with no reinstall step.
Drop the `-e` for a plain copy. To install straight from GitHub without a local
clone:

```bash
pip install "phy @ git+https://github.com/cortex-lab/phy.git"
```

### Kwik GUI dependencies

**The legacy Kwik GUI requires Python 3.10 or 3.11.** It depends on `klusta` and
`klustakwik2`, both unmaintained since 2018, and the `kwik` extra pins
`numpy>=1.23,<1.24` to keep the whole legacy stack on the numpy it was written
against (see below). numpy 1.23 publishes no cp312 wheels, so Python 3.12 and
3.13 are out for the Kwik GUI. Nothing here affects a plain `pip install phy`.

```bash
pip install "phy[kwik]"
```

On macOS and on Apple Silicon there are no `klustakwik2` wheels, and its
`setup.py` imports numpy at build time, so build isolation has to be disabled:

```bash
pip install "cython>=3.0"
pip install --no-build-isolation klustakwik2
pip install "phy[kwik]"
```

The `kwik` extra enforces three constraints, all of them caused by the two legacy
packages rather than by phy:

* `numpy>=1.23,<1.24` — the last numpy the legacy stack was written against. This
  keeps the entire Kwik surface on tested ground and gives one self-contained
  setup, at the cost of Python 3.12+ support. It is deliberately conservative:
  reclustering itself runs on newer numpy, and even numpy 2 works once two
  upstream call sites (`ndarray.tostring()` in `klusta/kwik/h5.py` and
  `klustakwik2/precomputations.py`, removed in numpy 2.0) are updated to
  `.tobytes()`. When those fixes are released upstream the cap can be relaxed.
* `setuptools<81` — `klusta/__init__.py` imports `pkg_resources`, removed in 81.
* `six` — `klusta` declares no dependencies of its own at all.

Under this extra, opening `.kwik` files, curating them, and the GUI's `recluster`
action (`shift+ctrl+K`, which re-runs KlustaKwik2) all work.

Note that `klustakwik2` must also be *compiled* against the numpy major version
it will run under: a later `pip install` that pulls a different numpy into the
environment can break it with
`ImportError: numpy.core.multiarray failed to import`.

## Quick start

Open the Template GUI on a spike sorting output directory containing `params.py`:

```bash
cd path/to/my/spikesorting/output
phy template-gui params.py
```

Other useful commands:

```bash
phy template-describe params.py
phy kwik-gui path/to/file.kwik
phy trace-gui path/to/raw.bin --sample-rate 30000 --dtype int16 --n-channels 384
```

## Available GUIs and commands

### Template GUI

Use the Template GUI for current template-based workflows such as KiloSort and Spyking Circus.

```bash
phy template-gui params.py
```

To inspect a dataset from the terminal without launching the GUI:

```bash
phy template-describe params.py
```

### Kwik GUI

The Kwik GUI is still available for legacy kwik datasets, but it is no longer the primary workflow.

```bash
phy kwik-gui path/to/file.kwik
```

### Trace GUI

The Trace GUI is still experimental and opens raw electrophysiology recordings directly.

```bash
phy trace-gui path/to/raw.bin --sample-rate 30000 --dtype int16 --n-channels 384
```

## Running phy from Python

You can also launch phy from Python or IPython, which can be useful for debugging or profiling:

```python
from phy.apps.template import template_gui

template_gui("params.py")
```

## Developer setup

To work on phy itself in a fresh checkout:

```bash
git clone git@github.com:cortex-lab/phy.git
cd phy
uv sync --dev
```

If you are working on phy together with a local checkout of `phylib`, install that checkout in editable mode:

```bash
git clone git@github.com:cortex-lab/phylib.git
cd phylib
pip install -e . --upgrade
```

## Troubleshooting and docs

* [Documentation](https://phy.readthedocs.io/en/latest/)
* [Release notes](https://phy.readthedocs.io/en/latest/release/)
* [Troubleshooting](https://phy.readthedocs.io/en/latest/troubleshooting/)
* [GitHub issues](https://github.com/cortex-lab/phy/issues)
* [Mailing list](https://groups.google.com/forum/#!forum/phy-users)
* [Code of Conduct](CODE_OF_CONDUCT.md)

## Credits

**phy** is developed and maintained by [Cyrille Rossant](https://cyrille.rossant.net).

* [International Brain Laboratory](https://internationalbrainlab.org)
* [Cortex Lab (UCL)](https://www.ucl.ac.uk/cortexlab/) ([Kenneth Harris](https://www.ucl.ac.uk/biosciences/people/harris-kenneth) and [Matteo Carandini](https://www.carandinilab.net/))

Contributors to the repository are:

* Maxime Beau
* [Alessio Buccino](https://github.com/alejoe91)
* Thad Czuba
* [Michael Economo](https://github.com/mswallac)
* Einsied
* [Cedric Gestes](https://github.com/cgestes)
* Yaroslav Halchenko
* [Max Hunter](https://iris.ucl.ac.uk/iris/browse/profile?upi=MLDHU99)
* [Shabnam Kadir](https://iris.ucl.ac.uk/iris/browse/profile?upi=SKADI56)
* [Zach McKenzie](https://github.com/zm711)
* Sam Minkowicz
* [Christopher Nolan](https://github.com/crnolan)
* [Jesús Peñaloza](https://github.com/jpenalozaa)
* [Luke Shaheen](https://github.com/LukeShaheen)
* [Martin Spacek](http://mspacek.github.io/)
* [Nick Steinmetz](http://www.nicksteinmetz.com/)
* Olivier Winter
* szapp
* ycanerol
