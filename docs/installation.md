# Installation

## Requirements

phy works better on computers with sufficient RAM, a recent graphics card, and an SSD to store data files.


## Dependencies

Main dependencies are NumPy, SciPy, PyQt5, pyopengl, joblib. The full list of dependencies is available in the `environment.yml` file.


## Instructions

Minimal installation instructions (to be completed):

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/).

2. Download the [environment file](https://raw.githubusercontent.com/cortex-lab/phy/dev/environment.yml).

3. Open a terminal and run:

    ```
    conda env create -f path/to/environment.yml
    ```

    Wait for the virtual environment to be created and the dependencies installed.

4. Activate the virtual environment: `conda activate phy2`
5. Run phy:
    ```
    cd path/to/my/spikesorting/output
    phy template-gui params.py
    ```
