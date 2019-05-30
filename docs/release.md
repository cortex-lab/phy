# Release notes

Upcoming version is phy 2.0.

## New views

* Raster view: spike trains of all clusters
* Template view: template waveforms of all clusters
* Cluster statistics:
    * ISI
    * Instantaneous firing rate
    * Template amplitude histogram
    * Write your own

## New features

* Split clusters in the amplitude view or in the template feature view, in addition to the feature view
* Cluster view:
    * Dynamically filter the list of clusters based on cluster metrics and labels
    * New default column: template waveform amplitude
* Correlogram view: show baseline firing rate and refractory period
* All views:
    * Add multiple views of the same type
    * Closable views
    * Toggle automatic update of views upon cluster selection
    * Axes
    * PNG screenshot
* Trace view, raster view, template view: color mapping for clusters based on any cluster metrics or labels


## Improvements

* Selection of "best" channels is now based on template waveform amplitude and probe geometry (this is the peak channel, plus all neighbor channels where the amplitude is higher than a fixed fraction of the peak channel template amplitude)
* Fix scaling issues in waveform view
* Menu reorganization
* Bug fixes with cluster labeling


## Internal changes

* Dropped Python 2 support (end of life early 2020), Python 3.7+
* Updated to PyQt5
* Improved OpenGL-based plotting API (based on a fork of glumpy instead of vispy)
* Dropped the phy-contrib repository
* Created a small phylib dependency with I/O code and non-graphical utilities, used by ibllib
* Moved the phy GitHub repository from kwikteam to cortex-lab organization
