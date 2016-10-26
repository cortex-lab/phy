# phy project

[![Build Status](https://img.shields.io/travis/kwikteam/phy.svg)](https://travis-ci.org/kwikteam/phy)
[![codecov.io](https://img.shields.io/codecov/c/github/kwikteam/phy.svg)](http://codecov.io/github/kwikteam/phy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/phy/badge/?version=latest)](https://readthedocs.org/projects/phy/?badge=latest)
[![PyPI release](https://img.shields.io/pypi/v/phy.svg)](https://pypi.python.org/pypi/phy)
[![GitHub release](https://img.shields.io/github/release/kwikteam/phy.svg)](https://github.com/kwikteam/phy/releases/latest)
[![Join the chat at https://gitter.im/kwikteam/phy](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/kwikteam/phy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[**phy**](https://github.com/kwikteam/phy) is an open source neurophysiological data analysis package in Python. It provides features for sorting, analyzing, and visualizing extracellular recordings made with high-density multielectrode arrays containing hundreds to thousands of recording sites.


## Overview

**phy** contains the following subpackages:

* **phy.cluster.manual**: an API for manual sorting, used to create graphical interfaces for neurophysiological data
* **phy.gui**: a generic API for creating desktop applications with PyQt.
* **phy.plot**: a generic API for creating high-performance plots with VisPy (using the graphics processor via OpenGL)

phy targets developers and doesn't provide any I/O code. It only provides Python routines to process and visualize data.


## phy-contrib

The [phy-contrib](https://github.com/kwikteam/phy-contrib) repo contains a set of plugins with integrated GUIs that work with dedicated automatic clustering software. Currently it provides:

* **KwikGUI**: a manual sorting GUI that works with data processed with [**klusta**](http://klusta.readthedocs.org/en/latest/), an automatic clustering package.
* **TemplateGUI**: a manual sorting GUI that works with data processed with **Spyking Circus** and **KiloSort** (not released yet), which are template-matching-based spike sorting algorithms.


## Installation

**Note**: the installation instructions will be simplified soon.

1. Make sure that you have [**miniconda**](http://conda.pydata.org/miniconda.html) installed. You can choose the Python 3.5 64-bit version for your operating system (Linux, Windows, or OS X).
2. **Download the [environment file](https://raw.githubusercontent.com/kwikteam/phy/master/installer/environment.yml).**
3. **Open a terminal** (on Windows, `cmd`, not Powershell) in the directory where you saved the file and type:

    ```bash
    conda env create -n phy
    source activate phy  # omit the `source` on Windows
    pip install phy phycontrib
    ```
4. **Done**! Now, to use phy, you have to first type `source activate phy` in a terminal (omit the `source` on Windows), and then call `phy`.


### Updating the software

To get the latest version of the software, open a terminal and type:

```
source activate phy  # omit the `source` on Windows
pip install phy phycontrib --upgrade
```


## Links

* [User documentation of the Template GUI](http://phy-contrib.readthedocs.io/en/latest/template-gui/)
* [Developer documentation](http://phy.readthedocs.org/en/latest/) (work in progress)
* [Mailing list](https://groups.google.com/forum/#!forum/phy-users)
* [Sample data repository](http://phy.cortexlab.net/data/) (work in progress)


## Credits

**phy** is developed by [Cyrille Rossant](http://cyrille.rossant.net), [Shabnam Kadir](https://iris.ucl.ac.uk/iris/browse/profile?upi=SKADI56), [Dan Goodman](http://thesamovar.net/), [Max Hunter](https://iris.ucl.ac.uk/iris/browse/profile?upi=MLDHU99), and [Kenneth Harris](https://iris.ucl.ac.uk/iris/browse/profile?upi=KDHAR02), in the [Cortexlab](https://www.ucl.ac.uk/cortexlab), University College London.
