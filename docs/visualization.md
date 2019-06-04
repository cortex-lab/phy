# Interactive visualization of ephys data

phy provides two GUIs:

* the **Template GUI** for KiloSort/SpykingCircus datasets.
* the **Kwik GUI** for Kwik datasets, obtained with the klusta spike-sorting program.


## Opening a dataset in the GUI

*Note*: there may be a simpler procedure to open the GUI on Windows in the future.

To open the GUI on a given dataset, you need to use the command-line from the directory containing your dataset.

### KiloSort/SpykingCircus

Type `phy template-gui params.py` in the directory that contains the `params.py` file.

The dataset is made of a set of `.npy` files (`spike_times.npy`, `spike_clusters.npy`, and so on). There are also `.tsv` files for cluster-dependent data.

### Kwik/Klusta

Type `phy kwik-gui filename.kwik` in the directory that contains the `filename.kwik` file.


## General presentation of the GUI

The GUI is made of several parts:

* Menu bar (top)
* Dock widgets (main window)
    * Cluster view
    * Similarity view
    * Graphical views
* Status bar (bottom)

Dock widgets can be moved anywhere in the GUI. They can be closed as well. New views can be added from the `View` menu in the menu bar.


### Cluster view

The **Cluster view** shows the list of all clusters in your dataset.

#### Definition

A cluster is a given set of spikes, supposed to correspond to a single neuron. A cluster is identified by a unique integer, the **cluster id**. The cluster id is unique: when the cluster changes (i.e. some spikes are removed or added), the cluster id changes.

#### Cluster selection

You can click on one cluster to select it. Select multiple clusters by keeping **Control** or **Shift** pressed. Selected clusters are shown in the different graphical views (detailled below). Clustering actions (merge, split, move, label...) operate on selected clusters.

#### Cluster table

Default columns in the cluster view include the cluster id, best channel (channel with peak waveform amplitude), depth (mostly useful for Neuropixels probes), n_spikes. Click on a column to sort by the corresponding attribute. You can add custom columns (labels, see next page). Use the `:s` snippet to quickly sort by a given column.

#### Cluster group

Clusters found by spike sorting algorithms have different qualities. Some are genuine single units, others are mixtures of neurons, others are essentially made of artifacts. For historical reasons, the **cluster group** is one of:

* 0: `noise` (dark grey)
* 1: `mua` (multi-unit activity, light grey)
* 2: `good` (green)
* None: unsorted (white)

Rows are shown in different colors according to the cluster group.

#### Cluster filtering

You can filter the list of clusters shown in the cluster view, in the `filter` text box at the top of the cluster view. You can type a boolean expression using the column names as variables, and pressing `Enter`. You can also use the `:f` snippet. The syntax is Javascript. Here are a few examples:

```
group == 'good'
n_spikes > 10000
group != 'noise' && depth >= 1000
```


### Similarity view

The similarity view is very similar to the cluster view. It has an additional column: the **similarity**. It represents the similarity to clusters selected in the cluster view. As such, its contents change every time the cluster selection changes in the cluster view. By default, clusters in the similarity view are sorted by decreasing similarity.

The similarity score is obtained from the `similar_templates.npy` file.


## Graphical views

Graphical views constitute the most important part of the GUI. They represent different aspects of the selected clusters, and the corresponding spikes.

Views can be resized, moved around, tabbed in the GUI. You can close views that you don't need, add further views. You can also add multiple views of the same type. You can disable automatic updating of any view upon cluster selection.

You can pan (left-click and drag) and zoom (right-click and drag, mouse wheel) with the mouse in every view. Double-click to reset.


### Waveform view

This view shows the waveforms of a selection of spikes, on the relevant channels (based on amplitude and proximity to the peak waveform amplitude channel).

You can select a channel with **Control+click** (this impacts the feature view).

### Feature view

This view shows the principal component features of a selection of spikes in the selected clusters, on the relevant channels. The exact channels can be changed by control-clicking in the waveform view.

Background spikes from all clusters are shown in grey.

### Correlogram view

This view shows the autocorrelograms and cross-correlograms between all pairs of selected clusters.

The baseline firing rate is shown. You can also display horizontal lines for the refractory period (this is a parameter that can be changed).

### Trace view

This view shows the raw data traces across all channels, with spikes from the selected clusters as well. You can also choose to show spikes from *all* clusters, not just selected clusters.

### Amplitude view

This view shows the amplitude of a selection of spikes belonging to the selected clusters. The spike amplitudes are stored in `amplitudes.npy`.

### Cluster statistics view

This generic view shows histogram related to the selected clusters. Built-in statistics views include:

* Inter-spike intervals
* Instantenous firing-rate
* Template amplitude

### Raster view

This view shows a raster plot of *all* clusters. The order of the rows depends on the sort in the cluster view. This view also reacts to filtering in the cluster view.

### Template view

This view shows all templates. The position of the templates depends on the sort in the cluster view. This view also reacts to filtering in the cluster view.
