# phy documentation

phy is an open-source graphical interface for visualizing and manually curating large-scale
electrophysiology datasets. It is optimized for high-density probes, including Neuropixels, and
provides both an interactive curation workflow and Python extension points for specialized labs.

> **Current stable release:** phy 2.1.0. The source tree is developing phy 2.1.1. See the
> [release notes](release.md) and [changelog](changelog.md).

[![Template GUI](https://user-images.githubusercontent.com/1942359/74028054-c284b880-49a9-11ea-8815-1b7e727a8644.png)](https://user-images.githubusercontent.com/1942359/74028054-c284b880-49a9-11ea-8815-1b7e727a8644.png)

## Start here

New users should follow these pages in order:

1. [Install phy](installation.md) in a fresh Python environment.
2. [Prepare and validate a dataset](dataset.md).
3. Follow the [first-ten-minutes quickstart](quickstart.md).
4. Learn the [GUI selection model](gui.md) and
   [similarity-guided workflow](similarity.md).
5. Read how [saving and output files](outputs.md) work before curating valuable data.

## Supported workflows

The **Template GUI** is the primary workflow for template-based spike sorters such as
[KiloSort](https://github.com/MouseLand/Kilosort) and
[SpyKING CIRCUS](https://spyking-circus.readthedocs.io/). It reads NumPy arrays, TSV metadata,
and `params.py`.

The **Kwik GUI** remains available for legacy klusta/Kwik datasets but is no longer the primary
maintained workflow. The experimental **Trace GUI** opens a continuous raw recording directly.
See the [command-line reference](cli.md) for all entry points.

## Customize and extend phy

Common settings—including spike sampling, caches, state, and keyboard shortcuts—are covered under
[Configuration and customization](configuration.md). Plugin authors and analysts can continue
with the [power-user guide](advanced.md), [plugin examples](plugins.md), and
[Python API](api.md).
