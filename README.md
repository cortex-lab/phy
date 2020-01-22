# phy: interactive visualization and manual spike sorting of large-scale ephys data

[![Build Status](https://travis-ci.org/cortex-lab/phy.svg?branch=dev)](https://travis-ci.org/cortex-lab/phy)
[![codecov.io](https://img.shields.io/codecov/c/github/cortex-lab/phy.svg)](http://codecov.io/github/cortex-lab/phy?branch=dev)
[![Documentation Status](https://readthedocs.org/projects/phy/badge/?version=latest)](https://phy.readthedocs.io/en/latest/?badge=latest)
[![GitHub release](https://img.shields.io/github/release/cortex-lab/phy.svg)](https://github.com/cortex-lab/phy/releases/latest)
<!-- [![PyPI release](https://img.shields.io/pypi/v/phy.svg)](https://pypi.python.org/pypi/phy) -->

[**phy**](https://github.com/cortex-lab/phy) is an open-source Python library providing a graphical user interface for visualization and manual curation of large-scale electrophysiological data. It is optimized for high-density multielectrode arrays containing hundreds to thousands of recording sites (mostly [Neuropixels probes](https://www.ucl.ac.uk/neuropixels/)).

[![phy 2.0b1 screenshot](https://user-images.githubusercontent.com/1942359/72895319-5b1a0800-3d1d-11ea-865e-26b09e9f4239.png)](https://user-images.githubusercontent.com/1942359/72895319-5b1a0800-3d1d-11ea-865e-26b09e9f4239.png)




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

**phy** is developed by [Cyrille Rossant](http://cyrille.rossant.net) in the [Cortex Lab](https://www.ucl.ac.uk/cortexlab/) at UCL (led by [Kenneth Harris](https://www.ucl.ac.uk/biosciences/people/harris-kenneth) and [Matteo Carandini](https://www.carandinilab.net/)).

Contributors to the repository are:

* [Alessio Buccino](https://github.com/alejoe91)
* [Michael Economo](https://github.com/mswallac)
* [Cedric Gestes](https://github.com/cgestes)
* [Dan Goodman](http://thesamovar.net/)
* [Max Hunter](https://iris.ucl.ac.uk/iris/browse/profile?upi=MLDHU99)
* [Shabnam Kadir](https://iris.ucl.ac.uk/iris/browse/profile?upi=SKADI56)
* [Christopher Nolan](https://github.com/crnolan)
* [Martin Spacek](http://mspacek.github.io/)
* [Nick Steinmetz](http://www.nicksteinmetz.com/)
