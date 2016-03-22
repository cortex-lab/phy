# phy project

[![Build Status](https://img.shields.io/travis/kwikteam/phy.svg)](https://travis-ci.org/kwikteam/phy)
[![Build Status](https://img.shields.io/appveyor/ci/kwikteam/phy.svg)](https://ci.appveyor.com/project/kwikteam/phy/)
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

phy doesn't provide any I/O code. It only provides Python routines to process and visualize data.


## phy-contrib

The [phy-contrib](https://github.com/kwikteam/phy-contrib) repo contains a set of plugins with integrated GUIs that work with dedicated automatic clustering software. Currently it provides:

* **KwikGUI**: a manual sorting GUI that works with data processed with **klusta**, an automatic clustering package.
* **TemplateGUI**: a manual sorting GUI that works with data processed with **Spyking Circus** and **KiloSort** (not released yet), which are template-matching-based spike sorting algorithms.


## Getting started

You will find installation instructions and a quick start guide in the [documentation](http://phy.readthedocs.org/en/latest/) (work in progress).


## Links

* [Documentation](http://phy.readthedocs.org/en/latest/) (work in progress)
* [Mailing list](https://groups.google.com/forum/#!forum/phy-users)
* [Sample data repository](http://phy.cortexlab.net/data/) (work in progress)


## Credits

**phy** is developed by [Cyrille Rossant](http://cyrille.rossant.net), [Shabnam Kadir](https://iris.ucl.ac.uk/iris/browse/profile?upi=SKADI56), [Dan Goodman](http://thesamovar.net/), [Max Hunter](https://iris.ucl.ac.uk/iris/browse/profile?upi=MLDHU99), and [Kenneth Harris](https://iris.ucl.ac.uk/iris/browse/profile?upi=KDHAR02), in the [Cortexlab](https://www.ucl.ac.uk/cortexlab), University College London.
