# phy: interactive visualization and manual spike sorting of large-scale ephys data

[![Build Status](https://travis-ci.org/cortex-lab/phy.svg?branch=dev)](https://travis-ci.org/cortex-lab/phy)
[![codecov.io](https://img.shields.io/codecov/c/github/cortex-lab/phy.svg)](http://codecov.io/github/cortex-lab/phy?branch=dev)
[![Documentation Status](https://readthedocs.org/projects/phy/badge/?version=latest)](https://phy.readthedocs.io/en/latest/?badge=latest)
[![GitHub release](https://img.shields.io/github/release/cortex-lab/phy.svg)](https://github.com/cortex-lab/phy/releases/latest)
<!-- [![PyPI release](https://img.shields.io/pypi/v/phy.svg)](https://pypi.python.org/pypi/phy) -->

[**phy**](https://github.com/cortex-lab/phy) is an open-source Python library providing a graphical user interface for visualization and manual curation of large-scale electrophysiological data. It is optimized for high-density multielectrode arrays containing hundreds to thousands of recording sites (mostly [Neuropixels probes](https://www.ucl.ac.uk/neuropixels/)).

[![phy 2.0a1 screenshot](https://user-images.githubusercontent.com/1942359/61941399-b00bda00-af97-11e9-92d9-5b7308ed25ac.png)](https://user-images.githubusercontent.com/1942359/61941399-b00bda00-af97-11e9-92d9-5b7308ed25ac.png)


## Links

* [**Documentation**](http://phy.readthedocs.org/en/latest/)
* [Mailing list](https://groups.google.com/forum/#!forum/phy-users)


## User installation instructions

[**Click here to see the installation instructions.**](https://phy.readthedocs.io/en/latest/installation/)


## Developer installation instructions

Use these instructions if you're a **developer** who wants to contribute to phy. Assuming you have a scientific Python distribution like Anaconda with the most important dependencies installed (NumPy, SciPy, matplotlib, IPython, PyQt5...), do **in a fresh environment** (and NOT in an environment that already has phy 1.x installed):

```bash
git clone -b dev https://github.com/cortex-lab/phy.git
cd phy
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Credits

**phy** is developed by [Cyrille Rossant](http://cyrille.rossant.net).

Past and present contributors include:

* [Shabnam Kadir](https://iris.ucl.ac.uk/iris/browse/profile?upi=SKADI56)
* [Dan Goodman](http://thesamovar.net/)
* [Max Hunter](https://iris.ucl.ac.uk/iris/browse/profile?upi=MLDHU99)
* [Kenneth Harris](https://iris.ucl.ac.uk/iris/browse/profile?upi=KDHAR02)
