# Introduction

**NOTE: this documentation is for the development version of phy 2.0, available on GitHub, [cortex-lab/phy repository, dev branch](https://github.com/cortex-lab/phy/tree/dev). A release should be available soon. This is still a work-in-progress.**

[**phy**](https://github.com/cortex-lab/phy) is an open-source Python library providing a graphical user interface for visualization and manual curation of large-scale electrophysiological data. It is optimized for high-density multielectrode arrays containing hundreds to thousands of recording sites (mostly [Neuropixels probes](https://www.ucl.ac.uk/neuropixels/)).

[![phy 2.0a1 screenshot](https://user-images.githubusercontent.com/1942359/61941399-b00bda00-af97-11e9-92d9-5b7308ed25ac.png)](https://user-images.githubusercontent.com/1942359/61941399-b00bda00-af97-11e9-92d9-5b7308ed25ac.png)

## Spike sorting programs

phy can open datasets spike-sorted with the following programs:

* [KiloSort](https://github.com/MouseLand/Kilosort2/)
* [SpykingCircus](https://spyking-circus.readthedocs.io/en/latest/)
* [klusta](http://klusta.readthedocs.org/en/latest/)

KiloSort and SpykingCircus are spike sorting programs based on template matching. They use a file format based on `.npy` ([NumPy binary arrays](https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html)) and `.tsv` files (tab-separated files).

klusta is a legacy spike sorting program based on an automatic clustering algorithm. It uses the [kwik format](https://klusta.readthedocs.io/en/latest/kwik/#kwik-format) based on HDF5. While klusta and the kwik format are still supported by phy, they are no longer actively maintained.
