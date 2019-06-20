# Release notes

Current version is phy v2.0a1 (alpha 1). There may be further new features for v2.0.


## New views

* **Raster view**: spike trains of all clusters
* **Template view**: template waveforms of all clusters
* **Cluster statistics views** (histograms):
    * ISI
    * Instantaneous firing rate
    * Template amplitude histogram
    * Write your own
* **Spike attributes views**: a scatter view is automatically created for every `spike_somename.npy` containing 1D or 2D values per spike
* **IPython view**


## New features

* **Split clusters in the amplitude view or in the template feature view**, in addition to the feature view
* **Cluster view**:
    * Dynamically **filter** the list of clusters based on cluster metrics and labels (using JavaScript syntax)
    * Snippets to quickly **sort and filter** clusters
    * New default columns: mean firing rate, and template waveform amplitude
    * The styling can be customized with CSS in a plugin (see plugin examples in the documentation)
* **Amplitude view**:
    * Show histogram of amplitudes overlayed with the amplitudes
    * Support for multiple types of amplitudes (template waveform amplitude, raw waveform amplitude, feature amplitude)
    * Splitting is supported
* **Correlogram view**:
    * Show horizontal line for the baseline firing rate, and a customizable vertical line for the refractory period
* **Waveform view**:
    * Change the number of waveforms directly from the GUI
* **Feature view**:
    * One can now change the specification of the subplots in a plugin
* **All views**:
    * Add multiple views of the same type
    * Closable views
    * Toggle automatic update of views upon cluster selection
    * Axes (in most views)
    * PNG screenshot
    * Control+mouse wheel to change the scaling (in most views)
    * Change the number of "best" channels in the user configuration file
* **Trace view**:
    * Auto-update by default
* **Trace view, raster view, template view**: customizable color mapping for clusters based on any cluster metrics or labels
* **Template dataset**
    * Support **multiple raw data files** (virtually concatenated)
    * Support for **multiple channel shanks**: add a `channel_shanks.npy` file with shape `(n_channels,` ), with the shank integer index of every channel.


## Improvements

* A new file `cluster_info.tsv` is automatically saved, it contains all information from the cluster view.
* Minimal high-level data access API in the Template Model
* Improved performance, avoid blocking the GUI when loading large amounts of data
* Fix scaling issues in waveform view
* More efficient probe view
* Slightly different gray colors for noise and MUA clusters in the cluster view
* Menu reorganization
* Bug fixes with cluster labels
* Documentation rewritten from scratch, with many plugin and API examples


## Internal changes

* Support **Python 3.7+**, dropped Python 2 support (end of life early 2020)
* Updated to **PyQt5** from PyQt4, which is now unsupported
* Improved OpenGL-based plotting API (based on a fork of glumpy instead of vispy)
* Dropped the phy-contrib repository
* Created a small phylib dependency with I/O code and non-graphical utilities, used by ibllib
* Moved the phy GitHub repository from kwikteam to cortex-lab organization
