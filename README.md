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

Install phy in a fresh Python 3.10-3.13 environment:

```bash
python -m pip install --upgrade pip
pip install phy
```

This installs the GUI runtime dependencies as part of the main package.

phy runs on numpy 2 and Python 3.13.

### Reclustering

phy can recluster the selected clusters from their PC features, using ISO-SPLIT
(the algorithm MountainSort uses) or a Gaussian mixture. No spike detection is
re-run, so this needs no legacy dependencies: scikit-learn and isosplit are pure
Python and numpy 2 compatible, and it works on Python 3.13.

```bash
pip install "phy[kwik]"
```

Reclustering is **opt-in**: pass `--recluster` to enable it for a run. Without the
flag the GUI behaves exactly as before, so it never changes a normal session.

```bash
phy template-gui path/to/params.py              # no reclustering
phy template-gui path/to/params.py --recluster  # reclustering enabled
```

No configuration file is needed -- the plugin ships inside the phy package
(`phy/plugins/recluster.py`) and the flag enables it directly. It adds three
actions:

| shortcut | action |
| --- | --- |
| `alt+k` | ISO-SPLIT, which picks the number of subclusters itself |
| `shift+alt+k` | split into a number of subclusters you give, using a Gaussian mixture |
| `ctrl+alt+k` | ISO-SPLIT at a given dip threshold; lower it below 2 to split more readily |

If you want it on for every run without typing the flag, enable it in
`~/.phy/phy_config.py` instead:

```python
c = get_config()
c.TemplateGUI.plugins = ['ExampleReclusterPlugin']
```

Running `--recluster` without the optional dependencies installed leaves the
actions out and logs how to get them, rather than failing.

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

### The legacy Kwik GUI (`.kwik` files)

**Not supported.** The Kwik GUI needs `klusta` and `klustakwik2`, which are
unmaintained: klusta's spike detection uses `np.bool`/`np.int`/`np.object`
(removed in numpy 1.24) and klustakwik2 compiles against the numpy 1.x C API.
Together they cap an install at numpy 1.23 and Python 3.11, which is
incompatible with the numpy 2 and Python 3.13 support above.

The `kwik` extra therefore installs the reclustering dependencies only, not the
legacy stack. Use `phy template-gui params.py` on a Kilosort output directory.
If you need to open `.kwik` files, use an older phy release in a separate
Python 3.11 environment.

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
