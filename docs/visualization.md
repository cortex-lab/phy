# Interactive visualization of ephys data

phy provides two GUIs:

* the **Template GUI** for KiloSort/SpykingCircus datasets.
* the **Kwik GUI** for Kwik datasets, obtained with the klusta spike-sorting program.

These GUIs let you visualize ephys data that has already been spike-sorted. You can also refine the clustering manually if needed. Finally, you can use the GUI as a platform for interactive ephys data analysis. The **IPython view** lets you interact with the data interactively from within the GUI.


## Opening a dataset in the GUI

*Note*: there will be a simpler procedure to open the GUI on Windows in a future version.

To open the GUI on a given dataset, you need to use the command-line from the directory containing your dataset.

### KiloSort/SpykingCircus

Type `phy template-gui params.py` in the directory that contains the `params.py` file.

The dataset is made of a set of `.npy` files (`spike_times.npy`, `spike_clusters.npy`, and so on). There are also `.tsv` files for cluster-dependent data.

### Kwik/Klusta

Type `phy kwik-gui filename.kwik` in the directory that contains the `filename.kwik` file.



## General presentation of the GUI

*Note*: we focus here on the template GUI.

The GUI is made of several parts:

* Menu bar (top)
* Dock widgets (main window)
    * Cluster view
    * Similarity view
    * Graphical views
* Status bar (bottom)

[![Template GUI](https://user-images.githubusercontent.com/1942359/58665615-90f32200-8331-11e9-8403-9961c13b8f17.png)](https://user-images.githubusercontent.com/1942359/58665615-90f32200-8331-11e9-8403-9961c13b8f17.png)

Dock widgets can be moved anywhere in or outside of the GUI (floating mode). They can be closed as well. New views can be added from the `View` menu in the menu bar.

Use the menu, keyboard shortcuts, or snippets to trigger actions. Press `F1` to see the list of keyboard shortcuts and snippets.


### Cluster view

The **Cluster view** shows the list of all clusters in your dataset.

![Cluster view](https://user-images.githubusercontent.com/1942359/58951131-a97b9600-8790-11e9-9765-8b380522417e.png)

#### Definition

A cluster is a given set of spikes, supposed to correspond to a single neuron. A cluster is identified by a unique integer, the **cluster id**. The cluster id is unique: when the cluster changes (i.e. some spikes are removed or added), the cluster id changes.

#### Cluster selection

You can click on one cluster to select it. Select multiple clusters by keeping **Control** or **Shift** pressed. Selected clusters are shown in the different graphical views (detailled below). Clustering actions (merge, split, move, label...) operate on selected clusters.

![image](https://user-images.githubusercontent.com/1942359/58951169-bac4a280-8790-11e9-8e7b-5fa5410de152.png)

##### Colormaps

Selected clusters are assigned with a special color: blue for the first selected cluster, red for the second, yellow for the third, etc.

In addition to this temporary color mapping, there is also a notion of global color map that assigns a (relatively) unique color to all clusters. This is especially useful in views that can display many spikes from many non-selected clusters at once: the Trace view (when the `toggle_highlighted_spikes` option is enabled), the Raster view, and the Template view.

Several colormaps are provided by phy (linear, divergent, categorical...). You can choose which cluster attribute to use for the color mapping. For example, you can use a cluster color depending on the depth, the number of spikes, the waveform amplitude, etc.

#### Cluster table

Default columns in the cluster view include the cluster id, best channel (channel with peak waveform amplitude), depth (mostly useful for Neuropixels probes), n_spikes. Click on a column to sort by the corresponding attribute. You can add custom columns (labels, see next page). Use the `:s` snippet to quickly sort by a given column.

#### Cluster group

Clusters found by spike sorting algorithms have different qualities. Some are genuine single units, others are mixtures of neurons, others are essentially made of artifacts. For historical reasons, the **cluster group** is one of:

* 0: `noise` (dark grey)
* 1: `mua` (multi-unit activity, light grey)
* 2: `good` (green)
* None: unsorted (white)

Rows in the cluster view are shown in different colors according to the cluster group.

*To do*: customizable groups and associated colors.

#### Cluster filtering

You can filter the list of clusters shown in the cluster view, in the `filter` text box at the top of the cluster view. Type a boolean expression using the column names as variables, and press `Enter`. You can also use the `:f` snippet. The syntax is Javascript. Here are a few examples:

```
group == 'good'
n_spikes > 10000
group != 'noise' && depth >= 1000
```

![image](https://user-images.githubusercontent.com/1942359/58951225-d8920780-8790-11e9-8b3c-a048f929875b.png)


### Similarity view

The similarity view is very similar to the cluster view. It has an additional column: the **similarity**. It represents the similarity to clusters selected in the cluster view. As such, its contents change every time the cluster selection changes in the cluster view. By default, clusters in the similarity view are sorted by decreasing similarity.

The similarity score is obtained from the `similar_templates.npy` file.


## Graphical views

Graphical views constitute the most important part of the GUI. They represent different aspects of the selected clusters and the corresponding spikes.

Views can be resized, moved around, tabbed in the GUI. You can close views that you don't need, you can add new views. You can also add multiple views of the same type. You can disable automatic updating of any view upon cluster selection.

You can pan (left-click and drag) and zoom (right-click and drag, mouse wheel) with the mouse in every graphical view. Double-click to reset the view.


### Waveform view

This view shows the waveforms of a selection of spikes, on the relevant channels (based on amplitude and proximity to the peak waveform amplitude channel).

You can select a channel with **Control+click** (this impacts the feature view). You can change the scaling of the channel positions and the waveforms.

![image](https://user-images.githubusercontent.com/1942359/58951290-0414f200-8791-11e9-8858-096fa3f5dee4.png)

You can show: spike waveforms, mean spike waveforms, or template waveforms (`toggle_mean_waveforms` and `toggle_templates` actions).

#### Keyboard shortcuts and snippets

```
- decrease                                 ctrl+down
- extend_horizontally                      shift+right
- extend_vertically                        shift+up
- increase                                 ctrl+up
- narrow                                   ctrl+left
- shrink_horizontally                      shift+left
- shrink_vertically                        shift+down
- toggle_mean_waveforms                    m
- toggle_show_labels                       ctrl+l
- toggle_templates                         w
- toggle_waveform_overlap                  o
- widen                                    ctrl+right
```


### Feature view

This view shows the principal component features of a selection of spikes in the selected clusters, on the relevant channels. The exact channels can be changed by control-clicking in the waveform view. A, B, C... refer to the first, second, third... principal components.

![image](https://user-images.githubusercontent.com/1942359/58951435-6bcb3d00-8791-11e9-89e6-d2a901ee5c56.png)

The default subplot organization of the feature view is (x and y for each of the 4x4 subplots, 0 refers to first selected channel, 1 refers to second select channel):

```
time,0A 1A,0A   0B,0A   1B,0A
        time,1A 0B,1A   1B,1A
                time,0B 1B,0B
                        time,1B
```

Background spikes from all clusters are shown in grey.

*To do*: show something useful in the empty subplots.

#### Keyboard shortcuts and snippets

```
- clear_channels                           - (:clear_channels)
- decrease                                 ctrl+-
- increase                                 ctrl++
- toggle_automatic_channel_selection       c
```


### Template feature view

This view is only active when exactly two clusters are selected. It shows the `template_features.npy` file, which is created by KiloSort.

![image](https://user-images.githubusercontent.com/1942359/58952660-9ff42d00-8794-11e9-88ff-a31394ee9cea.png)



### Correlogram view

This view shows the autocorrelograms and cross-correlograms between all pairs of selected clusters.

The baseline firing rate is shown. You can also display horizontal lines for the refractory period.

![image](https://user-images.githubusercontent.com/1942359/58951508-9c12db80-8791-11e9-8cac-a6ca1ba7da9d.png)

#### Keyboard shortcuts and snippets

```
- set_bin                                  - (:cb)
- set_refractory_period                    - (:cr)
- set_window                               - (:cw)
- toggle_normalization                     n
```


### Trace view

This view shows the raw data traces across all channels, with spikes from the selected clusters as well. You can also choose to show spikes from *all* clusters, not just selected clusters.

![image](https://user-images.githubusercontent.com/1942359/58951569-bf3d8b00-8791-11e9-969b-9327a4f58811.png)

#### Keyboard shortcuts and snippets

```
- decrease                                 alt+down
- go_left                                  alt+left
- go_right                                 alt+right
- go_to                                    alt+t
- go_to_next_spike                         alt+pgdown
- go_to_previous_spike                     alt+pgup
- increase                                 alt+up
- narrow                                   alt++
- shift                                    - (:ts)
- toggle_highlighted_spikes                alt+s
- toggle_show_labels                       alt+l
- widen                                    alt+-
```


### Amplitude view

This view shows the amplitude of a selection of spikes belonging to the selected clusters. The spike amplitudes are stored in `amplitudes.npy`.

![image](https://user-images.githubusercontent.com/1942359/58951635-e98f4880-8791-11e9-8a80-7e25d04a0fb4.png)


### Cluster statistics view

This generic view shows histogram related to the selected clusters. Built-in statistics views include:

* Inter-spike intervals
* Instantenous firing-rate
* Template amplitude

![image](https://user-images.githubusercontent.com/1942359/58951704-193e5080-8792-11e9-873f-91a9115a9e7c.png)

#### Keyboard shortcuts and snippets

```
Keyboard shortcuts for TemplateGUI - AmplitudeHistogramView
- set_n_bins                               - (:an)
- set_x_max                                - (:am)

Keyboard shortcuts for TemplateGUI - FiringRateView
- set_n_bins                               - (:fn)
- set_x_max                                - (:fm)

Keyboard shortcuts for TemplateGUI - ISIView
- set_n_bins                               - (:in)
- set_x_max                                - (:im)
```


### Raster view

This view shows a raster plot of *all* clusters. The order of the rows depends on the sort in the cluster view. If filtering is enabled in the cluster view, only filtered in clusters are shown in the raster view.

![image](https://user-images.githubusercontent.com/1942359/58951801-599dce80-8792-11e9-83af-ba78f6a2437b.png)


#### Keyboard shortcuts and snippets

```
- decrease                                 ctrl+shift+-
- increase                                 ctrl+shift++
```


### Template view

This view shows all templates. The position of the templates depends on the sort in the cluster view. If filtering is enabled in the cluster view, only filtered in clusters are shown in the template view.

![image](https://user-images.githubusercontent.com/1942359/58951871-894cd680-8792-11e9-85bb-f14441a132b7.png)

#### Keyboard shortcuts and snippets

```
- decrease                                 ctrl+alt+-
- increase                                 ctrl+alt++
```


### Spike attribute view

A **spike attribute view** is a view automatically created for every `spike_somename.npy` file in the data directory, that contains as many 1D or 2D points as spikes. In other words, the array shape should be `(n_spikes,)` or `(n_spikes, 2)`:

* 1D array: the view shows time (x axis) versus the value (y axis) for every spike
* 2D array: the view shows the (x, y) values for every spike

In the following screenshot, a `spike_hello.npy` array containing `sin(spike_time)` was saved, and a `SpikeHelloView` was automatically created:

![image](https://user-images.githubusercontent.com/1942359/58956662-2a418e80-879f-11e9-8227-2c56db2965e4.png)

You can split clusters by drawing polygons in the spike attribute views, as in the feature, amplitude, and template feature views.


## IPython view

The **IPython view** is an interactive IPython console that runs in the GUI's process. It lets you interact with the data and the GUI interactively.

For convenience, the following variables are available in the GUI:

* `m`: the `TemplateModel` instance that represents the dataset.
* `c`: the `TemplateController` instance that links the model to the views.
* `s`: the `Supervisor` instance that handles the cluster and similarity views, the cluster assignments, the clustering actions, etc. The clustering process by itself (which spikes are assigned to which clusters) is managed by `s.clustering`, a `Clustering` instance.

![image](https://user-images.githubusercontent.com/1942359/58953022-96b79000-8795-11e9-9bdd-77523c1c099e.png)

You can use matplotlib to make quick plots in the IPython view, although it is better to write a custom view properly if you need to reuse it (see the developer section in this documentation).