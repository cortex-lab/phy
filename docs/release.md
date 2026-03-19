# Release notes

Current version is phy v2.1.0rc1 (release candidate 1) (11 Mar 2026).


## phy 2.1.0rc1

With substantial help from AI-assisted development, it has been possible to put time and effort
into this maintenance release for the current 2.x line.

`phy 2.1.0rc1` is focused on making the existing software work better for users. It is not meant
to bring in major new features or feature requests at this stage. More work will likely still be
needed based on user feedback during the release candidate period.

### Main points

* No changes to the dataset or file formats
* Dependency updates and packaging cleanup
* Replacement of a fragile legacy web-based GUI component with a Qt-native implementation
* Expected improvement on systems where the old embedded web component caused blank panes, white windows, or related display failures

### What to test

* Installation on current Linux, macOS and Windows environments
* GUI startup and rendering behavior
* Cluster selection and view updates
* Feature, waveform, amplitude, and trace views
* Remote desktop, Wayland, and GPU-specific setups
* Plugin-based workflows

### Compatibility notes

* Dataset formats are unchanged
* Some plugins relying on internal HTML or other web-based GUI components may need updates

### Notes for plugin maintainers

The main compatibility risk for plugins is on the GUI side. The legacy web-based component has
been replaced with a Qt-native implementation, so plugins depending on internal HTML or other
web-based GUI pieces may need to be updated.

Plugins using supported Python-side controller, event, or view APIs are more likely to keep
working unchanged, but they should still be tested.

### Testing window

Testing for `2.1.0rc1` is expected to stay open for at least the next couple of months before a final `2.1.0` release.

### Feedback

When reporting issues, please include:

* operating system
* Python version
* installation method
* local or remote session details
* whether plugins are in use
* a minimal error message or reproduction if possible


## Historical notes


### phy 2.0 beta 1


#### New views

* **Cluster scatter view**: a scatter plot of all clusters, on two user-defined dimensions (for example, depth vs firing rate). The marker size and colors can also depend on two additional user-defined dimensions.

    <img src="https://user-images.githubusercontent.com/1942359/74028901-c74a6c00-49ab-11ea-9ead-4917bf9d2770.png" style="width: 400px;">

* **Raster view**: spike trains of all clusters

    <img src="https://user-images.githubusercontent.com/1942359/74028911-d0d3d400-49ab-11ea-84b5-d3e13cf0c5c6.png" style="width: 400px;">

* **Template view**: template waveforms of all clusters

    <img src="https://user-images.githubusercontent.com/1942359/74028982-ec3edf00-49ab-11ea-9dbb-4399472ff3ec.png" style="width: 400px;">

* **Cluster statistics views** (histograms):
    * ISI
    * Instantaneous firing rate
    * Write your own as a plugin

    <img src="https://user-images.githubusercontent.com/1942359/74029046-17293300-49ac-11ea-805a-82d7eab03062.png" style="width: 400px;">

* **Spike attributes views**: a scatter view is automatically created for every `spike_somename.npy` containing 1D or 2D values per spike

    <img src="https://user-images.githubusercontent.com/1942359/58956662-2a418e80-879f-11e9-8227-2c56db2965e4.png" style="width: 400px;">

* **Trace image view**: a minimal trace view that shows a large textured image `(n_channels, n_samples)` instead of multiple polylines.

    <img src="https://user-images.githubusercontent.com/1942359/74029179-696a5400-49ac-11ea-8b55-71cf864391aa.png" style="width: 400px;">


* Improved **Amplitude view**: different types of spike amplitudes as a function of time, with histograms.

    <img src="https://user-images.githubusercontent.com/1942359/74029349-c6fea080-49ac-11ea-9723-3f432023ee4b.png" style="width: 400px;">


* **IPython view**: interact with the GUI and the data programmatically.

    <img src="https://user-images.githubusercontent.com/1942359/74029254-90c12100-49ac-11ea-8b8e-a4997910ffe5.png" style="width: 400px;">


#### New features

* **Split clusters in the amplitude view or in the template feature view**, in addition to the feature view
* **Cluster view**:
    * Dynamically **filter** the list of clusters based on cluster metrics and labels (using JavaScript syntax)
    * Snippets to quickly **sort and filter** clusters
    * New default columns: mean firing rate, and template waveform amplitude
    * The styling can be customized in a plugin (see plugin examples in the documentation)
* **Amplitude view**:
    * Show an histogram of amplitudes overlaid with the amplitudes
    * Support for multiple types of amplitudes (template waveform amplitude, raw waveform amplitude, feature amplitude)
    * Splitting is supported
    * Alt+click in the amplitude view to center the trace view to a certain time (position shown with the vertical yellow bar)
    * Show the time interval currently displayed in the trace view as a vertical yellow bar
* **Correlogram view**:
    * Horizontal line for the baseline firing rate
    * Customizable vertical line for the refractory period
* **Waveform view**:
    * Change the number of waveforms directly from the GUI
    * Higher-quality thicker mean and template waveforms
* **Feature view**:
    * One can now change the specification of the subplots in a plugin
* **All views**:
    * Add multiple views of the same type
    * Closable views
    * View menu
    * Toggle automatic update of views upon cluster selection
    * Easily take screenshots of individual views
    * Control bar at the top of every view, with customizable text and buttons (screenshot, toggle auto update, etc.)
    * Axes (in most views)
    * Higher-quality text with OpenGL
    * Control+wheel to change the scaling (in most views)
    * Alt+wheel to change the marker size
    * Change the number of "best" channels in the user configuration file
* **Trace view**:
    * Auto-update by default
* **Trace view, raster view, template view**:
    * Customizable color mapping for clusters based on any cluster metrics or labels
* **Probe view**:
    * Channel labels
* **Template model**
    * Support **multiple raw data files** (virtually concatenated)
    * Support for **multiple channel shanks**: add a `channel_shanks.npy` file with shape `(n_channels,` ), with the shank integer index of every channel.


#### Improvements

* A new file `cluster_info.tsv` is automatically saved, containing all information from the cluster view.
* Minimal high-level data access API in the Template Model
* Improved performance, avoid blocking the GUI when loading large amounts of data
* Fix scaling issues in waveform view
* More efficient probe view
* Slightly different gray colors for noise and MUA clusters in the cluster view
* Menu reorganization
* Toolbar with icons
* Bug fixes with cluster labels
* New `plugins/` folder in the repository, with many plugin examples
* Documentation rewritten from scratch, with many examples


#### Internal changes

* Support **Python 3.7+**, dropped Python 2.x support (reached End Of Life)
* Updated to **PyQt5** from PyQt4, which is now unsupported
* Improved OpenGL-based plotting API (based on a fork of glumpy instead of vispy)
* Dropped the phy-contrib repository
* Created **phylib**, a small dependency with I/O code and non-graphical utilities, also used by ibllib
* Moved the phy GitHub repository from kwikteam to cortex-lab organization


#### Notes for plugin maintainers

The following changes may affect phy plugins:

* The `add_view` and `view_actions_created` events have been removed.
* You should now use the new event `view_attached(view, gui)` that is emitted when a view is attached to the GUI.
