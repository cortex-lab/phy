# `phy` project

This is a data analysis framework for large-scale electrophysiological data. The primary focus is on multielectrode extracellular recordings: raw data, LFP, and spike sorting. The scope may be expanded later on.


## Highlights

* Pure Python 2/3 framework
* Flexible and extendable
* Should be primarily used by experimentalists who like to code (= not idiot-proof library)
* At first, the development version on GitHub (master branch) should be the main version people will use
* Users can submit their own code through pull requests
* Out-of-the-box IPython notebook integration
* Data visualization with d3.js and Vispy
* Agnostic to the file format: can work with kwik files, Buzsaki files, Neo files, etc.
* Compatible with Apache Spark.


## Focus

* Raw data visualization in the IPython notebook with Vispy and WebGL
* Entire spike sorting toolchain: spike detection, feature extraction, clustering, manual stage
* Entirely customizable: users should be able to provide their own code for any part of the analysis
* Use the IPython notebook widgets to create light GUI elements within the notebook

