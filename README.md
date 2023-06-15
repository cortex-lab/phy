# phy: interactive visualization and manual spike sorting of large-scale ephys data

[![Build Status](https://travis-ci.org/cortex-lab/phy.svg)](https://travis-ci.org/cortex-lab/phy)
[![codecov.io](https://img.shields.io/codecov/c/github/cortex-lab/phy.svg)](http://codecov.io/github/cortex-lab/phy)
[![Documentation Status](https://readthedocs.org/projects/phy/badge/?version=latest)](https://phy.readthedocs.io/en/latest/?badge=latest)
[![GitHub release](https://img.shields.io/github/release/cortex-lab/phy.svg)](https://github.com/cortex-lab/phy/releases/latest)
[![PyPI release](https://img.shields.io/pypi/v/phy.svg)](https://pypi.python.org/pypi/phy)


[**phy**](https://github.com/cortex-lab/phy) is an open-source Python library providing a graphical user interface for visualization and manual curation of large-scale electrophysiological data. It is optimized for high-density multielectrode arrays containing hundreds to thousands of recording sites (mostly [Neuropixels probes](https://www.ucl.ac.uk/neuropixels/)).

Phy provides two GUIs:

* **Template GUI** (recommended): for datasets sorted with KiloSort and Spyking Circus,
* **Kwik GUI** (legacy): for datasets sorted with klusta and klustakwik2.


[![phy 2.0b1 screenshot](https://user-images.githubusercontent.com/1942359/74028054-c284b880-49a9-11ea-8815-1b7e727a8644.png)](https://user-images.githubusercontent.com/1942359/74028054-c284b880-49a9-11ea-8815-1b7e727a8644.png)


## What's new

* [7 Sep 2021] Release of phy 2.0 beta 5, with some install and bug fixes
* [7 Feb 2020] Release of phy 2.0 beta 1, with many new views, new features, various improvements and bug fixes...


## Links

* [Documentation](http://phy.readthedocs.org/en/latest/)
* [Mailing list](https://groups.google.com/forum/#!forum/phy-users)


## Hardware requirements

It is recommanded to store the data on a SSD for performance reasons.

There are no specific GPU requirements as long as relatively recent graphics and OpenGL drivers are installed on the system.


## Installation instructions

Run the following commands in a terminal:

1. Create a new conda environment with the conda dependencies:

    ```
    conda create -n phy2 -y cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt pyqtwebengine pytest python qtconsole requests responses scikit-learn scipy traitlets
    ```

2. Activate the new conda environment with `conda activate phy2`

3. Install the development version of phy: `pip install git+https://github.com/cortex-lab/phy.git`

4. [OPTIONAL] If you plan to use the Kwik GUI, type `pip install klusta klustakwik2`

5. Phy should now be installed. Open the GUI on a dataset as follows (the phy2 environment should still be activated):

```bash
cd path/to/my/spikesorting/output
phy template-gui params.py
```


### Dealing with the error `ModuleNotFoundError: No module named 'PyQt5.QtWebEngineWidget`

In some environments, you might get an error message related to QtWebEngineWidget. Run the command `pip install PyQtWebEngine` and try launching phy again. This command should not run if the error message doesn't appear, as it could break the PyQt5 installation.


### Upgrading from phy 1 to phy 2

* Do not install phy 1 and phy 2 in the same Python environment.
* It is recommended to delete `~/.phy/*GUI/state.json` when upgrading.


### Developer instructions

To install the development version of phy in a fresh environment, do:

```bash
git clone git@github.com:cortex-lab/phy.git
cd phy
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
cd ..
git clone git@github.com:cortex-lab/phylib.git
cd phylib
pip install -e . --upgrade
```

### Troubleshooting

* [See a list of common issues.](https://phy.readthedocs.io/en/latest/troubleshooting/)
* [Raise a GitHub issue.](https://github.com/cortex-lab/phy/issues)


## Running phy from a Python script

In addition to launching phy from the terminal with the `phy` command, you can also launch it from a Python script or an IPython terminal. This may be useful when debugging or profiling. Here's a code example to copy-paste in a new `launch.py` text file within your data directory:

```
from phy.apps.template import template_gui
template_gui("params.py")
```


## Credits

**phy** is developed and maintained by [Cyrille Rossant](https://cyrille.rossant.net).

* [International Brain Laboratory](https://internationalbrainlab.org)
* [Cortex Lab (UCL)](https://www.ucl.ac.uk/cortexlab/) ([Kenneth Harris](https://www.ucl.ac.uk/biosciences/people/harris-kenneth) and [Matteo Carandini](https://www.carandinilab.net/)).

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
