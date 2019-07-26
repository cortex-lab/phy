# Installation

## Requirements

phy works better on computers with sufficient RAM, a recent graphics card, and an SSD to store data files.


## Dependencies

Main dependencies are NumPy, SciPy, PyQt5, pyopengl, joblib. The full list of dependencies is available in the `environment.yml` file.

## Upgrading from phy 1 to phy 2

* Never install phy 1 and phy 2 in the same conda environment.
* It is recommended to delete `~/.phy/TemplateGUI/state.json` when upgrading.


## Instructions

Minimal installation instructions (to be completed):

1. Install [Anaconda](https://www.anaconda.com/distribution/#download-section).

2. Open a terminal and type:

    ```
    conda create -n phy2 python pip numpy matplotlib scipy h5py pyqt cython -y
    conda activate phy2
    pip install colorcet pyopengl qtconsole requests traitlets tqdm joblib click mkdocs PyQtWebEngine
    pip install git+https://github.com/cortex-lab/phy.git@dev
    pip install git+https://github.com/cortex-lab/phylib.git
    ```

3. Phy should now be installed. Open the GUI on a dataset as follows (the phy2 environment should still be activated):

    ```
    cd path/to/my/spikesorting/output
    phy template-gui params.py
    ```
