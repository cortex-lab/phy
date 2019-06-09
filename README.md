# phy: interactive visualization and manual spike sorting of large-scale ephys data

[![Build Status](https://img.shields.io/travis/cortex-lab/phy.svg)](https://travis-ci.org/cortex-lab/phy)
[![codecov.io](https://img.shields.io/codecov/c/github/cortex-lab/phy.svg)](http://codecov.io/github/cortex-lab/phy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/phy/badge/?version=latest)](https://readthedocs.org/projects/phy/?badge=latest)
[![PyPI release](https://img.shields.io/pypi/v/phy.svg)](https://pypi.python.org/pypi/phy)
[![GitHub release](https://img.shields.io/github/release/cortex-lab/phy.svg)](https://github.com/cortex-lab/phy/releases/latest)
[![Join the chat at https://gitter.im/cortex-lab/phy](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/cortex-lab/phy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[**phy**](https://github.com/cortex-lab/phy) is an open-source Python library providing a graphical user interface for visualization and manual curation of large-scale electrophysiological data. It is optimized for high-density multielectrode arrays containing hundreds to thousands of recording sites (mostly Neuropixels probes).

[![phy 2.0a0 screenshot](https://user-images.githubusercontent.com/1942359/58665615-90f32200-8331-11e9-8403-9961c13b8f17.png)](https://user-images.githubusercontent.com/1942359/58665615-90f32200-8331-11e9-8403-9961c13b8f17.png)

## Quick installation
1. Install [anaconda](https://docs.anaconda.com/anaconda/install/)
2. Download the [environment file](https://github.com/cortex-lab/phy/blob/dev/environment.yml)
3. Open a terminal and run:
`conda env create -f path/to/environment.yml`.\
Wait for the virtual environment to be created and the dependencies installed...
4. Activate the virtual environment: conda activate phy2
5. Run phy:
```
cd path/to/my/spikesorting/output
phy template-gui params.py
```

## Links

* [Documentation](http://phy.readthedocs.org/en/latest/)
* [Mailing list](https://groups.google.com/forum/#!forum/phy-users)


## Credits

**phy** is developed by [Cyrille Rossant](http://cyrille.rossant.net).

Past and present contributors include:

* [Shabnam Kadir](https://iris.ucl.ac.uk/iris/browse/profile?upi=SKADI56)
* [Dan Goodman](http://thesamovar.net/)
* [Max Hunter](https://iris.ucl.ac.uk/iris/browse/profile?upi=MLDHU99)
* [Kenneth Harris](https://iris.ucl.ac.uk/iris/browse/profile?upi=KDHAR02)
