# Plugin examples

In this section, we give many examples of plugins.

## Getting started

### How to use a plugin

1. Create a `myplugins.py` file in `~/.phy/plugins/` and copy-paste the code from [a plugin example on the GitHub repository](https://github.com/cortex-lab/phy/tree/master/plugins)
2. Edit `~/.phy/phy_config.py`, and specify the plugin names to load in the GUI:

```python
c.TemplateGUI.plugins = ['ExampleClusterStatsPlugin']  # list of plugin names to load in the TemplateGUI
```

The exact name of `myplugins.py` doesn't matter, since all Python scripts found in the plugin directories are automatically loaded by phy.

**Note**: there are three different concepts:

1. **Plugin directories**: the `~/.phy/phy_config.py` file contains a line like `c.Plugins.dirs = [r'~/.phy/plugins']` which is a list of directories that contain one or more **plugin files**.
2. **Plugin files**: a plugin file is any Python script that is found in the plugin directories. It contains one or more definitions of **plugin classes**.
3. **Plugin classes**: it is a Python class deriving from `IPlugin` which implements the logic of a plugin.

The `~/.phy/phy_config` file defines (1) the paths to the plugin directories (`c.Plugins.dirs`), and (2) the list of all plugin class names that needs to be loaded, and that are implemented in plugin files found in the plugin directories (`c.TemplateGUI.plugins`).

The idea is that one could *install* many plugins by putting the code in a plugin directory, but may not want to *activate* all of them every time.


### How to upgrade plugins from phy 1.0

Here are some things to know if you want to upgrade plugins to the latest v2.0 version of phy.

1. The event system has changed slightly. Replace:

```python
# DON'T USE. This was for phy 1.x, it does not work in phy 2.x
@myobject.connect
def on_event(arg):
    pass
```

by

```python
# USE THIS in phy 2.x.
from phy import connect

# 1. You can filter the events by sender. Here, this function is only called when the event
# is sent by `myobject`. If you don't add the `sender=` option, the function will be called
# for all events of that type sent by any object.
@connect(sender=myobject)
def on_eventname(sender, arg):
    # 2. The first positional parameter is always `sender` in event callbacks. It is mandatory.
    pass
```

2. Some controller methods have been renamed. See the [API documentation](api.md) for more details.

3. Make sure the deprecated package `phycontrib` is not loaded anywhere in your plugins, which could lead to conflicts. You should even make sure it is not installed in your phy2 environment.

4. Look at the plugin examples. They are good starting points to port your plugins. For example, there are example plugins for changing the number of spikes in the views, implementing custom recluster actions, adding custom matplotlib views, using custom cluster metrics and statistics, etc.


## Hello world

Here is how to write a simple plugin that displays a message every time there's a new cluster assignment:

```python
# import from plugins/hello.py
"""Hello world plugin."""

from phy import IPlugin, connect


class ExampleHelloPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect(sender=controller)
        def on_gui_ready(sender, gui):
            """This is called when the GUI and all objects are fully loaded.
            This is to make sure that controller.supervisor is properly defined.
            """

            @connect(sender=controller.supervisor)
            def on_cluster(sender, up):
                """This is called every time a cluster assignment or cluster group/label
                changes."""
                print("Clusters update: %s" % up)

```

This displays the following in the console, for example:

```
Clusters update: <merge [212, 299] => [305]>
Clusters update: <metadata_group [305] => good>
Clusters update: <metadata_neurontype [81, 82] => interneuron>
```


## Changing the number of spikes in views

You can customize the maximum number of spikes in the different views as follows (the example below shows the default values):

```python
# import from plugins/n_spikes_views.py
"""Show how to change the number of displayed spikes in each view."""

from phy import IPlugin


class ExampleNspikesViewsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Feel free to keep below just the values you need to change."""
        controller.n_spikes_waveforms = 250
        controller.batch_size_waveforms = 10
        controller.n_spikes_features = 2500
        controller.n_spikes_features_background = 2500
        controller.n_spikes_amplitudes = 2500
        controller.n_spikes_correlograms = 100000

        # Number of "best" channels kept for displaying the waveforms.
        controller.model.n_closest_channels = 12

        # The best channels are selected among the N closest to the best (peak) channel if their
        # mean amplitude is greater than this fraction of the peak amplitude on the best channel.
        # If zero, just the N closest channels are kept as the best channels.
        controller.model.amplitude_threshold = 0

```

*Note*: you need to manually delete the `.phy` subdirectory within your data directory when changing these parameters, otherwise errors will happen in the GUI.


## Customizing the default font size

In this plugin, we show how to change the default font size in the text visuals.

```python
# import from plugins/font_size.py
"""Show how to change the default text font size."""

from phy import IPlugin
from phy.plot.visuals import TextVisual


class ExampleFontSizePlugin(IPlugin):
    def attach_to_controller(self, controller):
        # Smaller font size than the default (6).
        TextVisual.default_font_size = 4.

```


## Customizing the columns of the cluster view

In this plugin, we show how to change the columns shown in the cluster and similarity views.

![image](https://user-images.githubusercontent.com/1942359/62779931-cac07180-bab4-11e9-967c-c2be0304b068.png)

```python
# import from plugins/custom_columns.py
"""Show how to customize the columns in the cluster and similarity views."""

from phy import IPlugin, connect


class ExampleCustomColumnsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_controller_ready(sender):
            controller.supervisor.columns = ['id', 'n_spikes']

```


## Defining a custom cluster metrics

In addition to cluster labels that you can create and modify in the cluster view, you can also define **cluster metrics**.

A cluster metrics is a function that assigns a scalar to every cluster. For example, any algorithm computing a kind of cluster quality is a cluster metrics.

You can define your own cluster metrics in a plugin. The values are then shown in a new column in the cluster view, and recomputed automatically when changing cluster assignments.

You can use the controller's methods to access the data. In rare cases, you may need to access the model directly (via `controller.model`).

For example, here is a custom cluster metrics that shows the mean inter-spike interval.

![image](https://user-images.githubusercontent.com/1942359/59463223-cf0a3e80-8e25-11e9-990f-e3e2dc9418a0.png)

```python
# import from plugins/cluster_metrics.py
"""Show how to add a custom cluster metrics."""

import numpy as np
from phy import IPlugin


class ExampleClusterMetricsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Note that this function is called at initialization time, *before* the supervisor is
        created. The `controller.cluster_metrics` items are then passed to the supervisor when
        constructing it."""

        def meanisi(cluster_id):
            t = controller.get_spike_times(cluster_id).data
            return np.diff(t).mean() if len(t) >= 2 else 0

        # Use this dictionary to define custom cluster metrics.
        # We memcache the function so that cluster metrics are only computed once and saved
        # within the session, and also between sessions (the memcached values are also saved
        # on disk).
        controller.cluster_metrics['meanisi'] = controller.context.memcache(meanisi)

```


## Writing a custom cluster similarity metrics

The similarity metrics measures the similarity between two clusters. It is used to get, in the similarity view, the list of clusters most similar to the clusters currently selected in the cluster view.

In the following example, we define a custom cluster similarity metrics based on a dot product between the mean waveforms.

![image](https://user-images.githubusercontent.com/1942359/60594697-d0d07d80-9da5-11e9-929f-d76433e444d2.png)

```python
# import from plugins/custom_similarity.py
"""Show how to add a custom similarity measure."""

from operator import itemgetter
import numpy as np

from phy import IPlugin
from phy.apps.template import from_sparse


def _dot_product(mw1, c1, mw2, c2):
    """Compute the L2 dot product between the mean waveforms of two clusters, given in sparse
    format."""

    mw1 = mw1[0, ...]  # first dimension has only 1 element.
    mw2 = mw2[0, ...]
    assert mw1.ndim == 2  # (n_samples, n_channels_loc_1)
    assert mw2.ndim == 2  # (n_samples, n_channels_loc_2)

    # We normalize the waveforms.
    mw1 /= np.sqrt(np.sum(mw1 ** 2))
    mw2 /= np.sqrt(np.sum(mw2 ** 2))

    # We find the union of the channel ids for both clusters so that we can convert from sparse
    # to dense format.
    channel_ids = np.union1d(c1, c2)

    # We directly return 0 if the channels of the two clusters are disjoint.
    if not len(np.intersect1d(c1, c2)):
        return 0

    # We tile the channels so as to use `from_sparse()`.
    c1 = np.tile(c1, (mw1.shape[0], 1))
    c2 = np.tile(c2, (mw2.shape[0], 1))

    # We convert from sparse to dense format in order to compute the distance.
    mw1 = from_sparse(mw1, c1, channel_ids)  # (n_samples, n_channel_locs_common)
    mw2 = from_sparse(mw2, c2, channel_ids)  # (n_samples, n_channel_locs_common)

    # We compute the dot product.
    return np.sum(mw1 * mw2)


class ExampleSimilarityPlugin(IPlugin):
    def attach_to_controller(self, controller):

        # We cache this function in memory and on disk.
        @controller.context.memcache
        def mean_waveform_similarity(cluster_id):
            """This function returns a list of pairs `(other_cluster_id, similarity)` sorted
            by decreasing similarity."""

            # We get the cluster's mean waveforms.
            mw = controller._get_mean_waveforms(cluster_id)
            mean_waveforms, channel_ids = mw.data, mw.channel_ids

            assert mean_waveforms is not None

            out = []
            # We go through all clusters except the currently selected one.
            for cl in controller.supervisor.clustering.cluster_ids:
                if cl == cluster_id:
                    continue
                mw = controller._get_mean_waveforms(cl)
                assert mw is not None
                # We compute the dot product between the current cluster and the other cluster.
                d = _dot_product(mean_waveforms, channel_ids, mw.data, mw.channel_ids)
                out.append((cl, d))  # convert from distance to similarity with a minus sign

            # We return the similar clusters by decreasing similarity.
            return sorted(out, key=itemgetter(1), reverse=True)

        # We add the similarity function.
        controller.similarity_functions['mean_waveform'] = mean_waveform_similarity

        # We set the similarity function to the newly-defined one.
        controller.similarity = 'mean_waveform'

```


## Writing a custom cluster statistics view

The ISI view, firing rate view, and amplitude histogram view all share the same characteristics. These views show histograms of cluster-dependent values. The number of bins and the maximum bin can be customized. All of these views derive from the **HistogramView**, a generic class for cluster statistics that can be displayed as histograms.

In the following example, we define a custom cluster statistics views using the PC features.

```python
# import from plugins/cluster_stats.py
"""Show how to add a custom cluster histogram view showing cluster statistics."""

from phy import IPlugin, Bunch
from phy.cluster.views import HistogramView


class FeatureHistogramView(HistogramView):
    """Every view corresponds to a unique view class, so we need to subclass HistogramView."""
    n_bins = 100  # default number of bins
    x_max = .1  # maximum value on the x axis (maximum bin)
    alias_char = 'fh'  # provide `:fhn` (set number of bins) and `:fhm` (set max bin) snippets


class ExampleClusterStatsPlugin(IPlugin):
    def attach_to_controller(self, controller):

        def feature_histogram(cluster_id):
            """Must return a Bunch object with data and optional x_max, plot, text items.

            The histogram is automatically computed by the view, this function should return
            the original data used to compute the histogram, rather than the histogram itself.

            """
            return Bunch(data=controller.get_features(cluster_id).data)

        def create_view():
            """Create and return a histogram view."""
            return FeatureHistogramView(cluster_stat=feature_histogram)

        # Maps a view name to a function that returns a view
        # when called with no argument.
        controller.view_creator['FeatureHistogram'] = create_view

```

![image](https://user-images.githubusercontent.com/1942359/58968835-fd00da80-87b6-11e9-8218-5adfc6e22a78.png)


## Customizing the styling of the cluster view

The cluster view is written in HTML/CSS/Javascript. The styling can be customized in a plugin as follows.

In this example, we change the text color of "good" clusters in the cluster view.

![image](https://user-images.githubusercontent.com/1942359/59463245-dd585a80-8e25-11e9-9fe3-56aa4c3733c7.png)

```python
# import from plugins/cluster_view_styling.py
"""Show how to customize the styling of the cluster view with CSS."""

from phy import IPlugin
from phy.cluster.supervisor import ClusterView


class ExampleClusterViewStylingPlugin(IPlugin):
    def attach_to_controller(self, controller):
        # We add a custom CSS style to the ClusterView.
        ClusterView._styles += """

            /* This CSS selector represents all rows for good clusters. */
            table tr[data-group='good'] {

                /* We change the text color. Many other CSS attributes can be changed,
                such as background-color, the font weight, etc. */
                color: red;
            }

        """

```


## Adding a new action

In this example, we create a new action in the file menu, with keyboard shortcut `a`, to display a message in the status bar. We create another action to select the first N clusters in the cluster view, where N is a parameter that the user can type in a prompt dialog.

![image](https://user-images.githubusercontent.com/1942359/59464230-2b6e5d80-8e28-11e9-804d-b94a57920f32.png)

```python
# import from plugins/action_status_bar.py
"""Show how to create new actions in the GUI.

The first action just displays a message in the status bar.

The second action selects the first N clusters, where N is a parameter that is entered by
the user in a prompt dialog.

"""

from phy import IPlugin, connect


class ExampleActionPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):

            # Add a separator at the end of the File menu.
            # Note: currently, there is no way to add actions at another position in the menu.
            gui.file_actions.separator()

            # Add a new action to the File menu.
            @gui.file_actions.add(shortcut='a')  # the keyboard shortcut is A
            def display_message():
                """Display Hello world in the status bar."""
                # This docstring will be displayed in the status bar when hovering the mouse over
                # the menu item.

                # We update the text in the status bar.
                gui.status_message = "Hello world"

            # We add a separator at the end of the Select menu.
            gui.select_actions.separator()

            # Add an action to a new submenu called "My submenu". This action displays a prompt
            # dialog with the default value 10.
            @gui.select_actions.add(
                submenu='My submenu', shortcut='ctrl+c', prompt=True, prompt_default=lambda: 10)
            def select_n_first_clusters(n_clusters):

                # All cluster view methods are called with a callback function because of the
                # asynchronous nature of Python-Javascript interactions in Qt5.
                @controller.supervisor.cluster_view.get_ids
                def get_cluster_ids(cluster_ids):
                    """This function is called when the ordered list of cluster ids is returned
                    by the Javascript view."""

                    # We select the first n_clusters clusters.
                    controller.supervisor.select(cluster_ids[:n_clusters])

```


## Creating a custom split action

In this example, we show how to write a custom split action, where one can split a cluster in two based on the template amplitudes, using the K-means algorithm.

![image](https://user-images.githubusercontent.com/1942359/59920302-65dc8980-942a-11e9-8414-2976fa59fa82.png)

```python
# import from plugins/custom_split.py
"""Show how to write a custom split action."""

from phy import IPlugin, connect


def k_means(x):
    """Cluster an array into two subclusters, using the K-means algorithm."""
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=2).fit_predict(x)


class ExampleCustomSplitPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='s')
            def custom_split():
                """Split using the K-means clustering algorithm on the template amplitudes
                of the first cluster."""

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Note that we need load_all=True to load all spikes from the selected clusters,
                # instead of just the selection of them chosen for display.
                bunchs = controller._amplitude_getter(cluster_ids, name='template', load_all=True)

                # We get the spike ids and the corresponding spike template amplitudes.
                # NOTE: in this example, we only consider the first selected cluster.
                spike_ids = bunchs[0].spike_ids
                y = bunchs[0].amplitudes

                # We perform the clustering algorithm, which returns an integer for each
                # subcluster.
                labels = k_means(y.reshape((-1, 1)))
                assert spike_ids.shape == labels.shape

                # We split according to the labels.
                controller.supervisor.actions.split(spike_ids, labels)

```

## Adding a new action

In this example, we create a new snippet to filter clusters with a firing rate above than a specified threshold. For example, typing `:fr 10` displays only the clusters with a firing rate higher than 10 spk/s in the cluster view. Pressing `Esc` clears the filter.

```python
# import from plugins/filter_action.py
"""Show how to create a filter snippet for the cluster view.

Typing `:fr 10` automatically shows only the clusters that have a firing rate higher than 10 spk/s.

"""

from phy import IPlugin, connect


class ExampleFilterFiringRatePlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            @gui.view_actions.add(alias='fr')  # corresponds to `:fr` snippet
            def filter_firing_rate(rate):
                """Filter clusters with the firing rate."""
                controller.supervisor.filter('fr > %.1f' % float(rate))

```


## Adding new buttons in the view title bars

In this example, we show how to add a button in the title bar of the WaveformView. Pressing this button switches the current type of waveforms displayed.

```python
# import from plugins/custom_button.py
"""Show how to add custom buttons in a view's title bar."""

from phy import IPlugin, connect
from phy.cluster.views import WaveformView


class ExampleCustomButtonPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_view_attached(view, gui):
            if isinstance(view, WaveformView):

                # view.dock is a DockWidget instance, it has methods such as add_button(),
                # add_checkbox(), and set_status().

                # The icon unicode can be found at https://fontawesome.com/icons?d=gallery
                @view.dock.add_button(icon='f105')
                def next_waveforms_type(checked):
                    # The checked argument is only used with buttons `checkable=True`
                    view.next_waveforms_type()

```


## Saving cluster metadata in a TSV file

In this example, we show how to save the best channel of all clusters in a TSV file.

```python
# import from plugins/cluster_metadata.py
"""Show how to save the best channel of every cluster in a cluster_channel.tsv file when saving.

Note: this information is also automatically stored in `cluster_info.tsv` natively in phy,
along with all values found in the GUI cluster view.

"""

import logging

from phy import IPlugin, connect
from phylib.io.model import save_metadata

logger = logging.getLogger('phy')


class ExampleClusterMetadataPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):

            @connect(sender=gui)
            def on_request_save(sender):
                """This function is called whenever the Save action is triggered."""

                # We get the filename.
                filename = controller.model.dir_path / 'cluster_channel.tsv'

                # We get the list of all clusters.
                cluster_ids = controller.supervisor.clustering.cluster_ids

                # Field name used in the header of the TSV file.
                field_name = 'channel'

                # NOTE: cluster_XXX.tsv files are automatically loaded in phy, displayed
                # in the cluster view, and interpreted as cluster labels, *except* if their
                # name conflicts with an existing built-in column in the cluster view.
                # This is the case here, because there is a default channel column in phy.
                # Therefore, the TSV file is properly saved, but it is not displayed in the
                # cluster view as the information is already shown in the built-in channel column.
                # If you want this file to be loaded in the cluster view, just use another
                # name that is not already used, like 'best_channel'.

                # Dictionary mapping cluster_ids to the best channel id.
                metadata = {
                    cluster_id: controller.get_best_channel(cluster_id)
                    for cluster_id in cluster_ids}

                # Save the metadata file.
                save_metadata(filename, field_name, metadata)
                logger.info("Saved %s.", filename)

```


## Injecting custom variables in the IPythonView

In this example, we show how to inject variables into the IPython console namespace. Specifically, we inject the first waveform view with the variable name `wv`.

```python
# import from plugins/ipython_view.py
"""Show how to injet specific Python variables in the IPython view."""

from phy import IPlugin, connect
from phy.cluster.views import WaveformView
from phy.gui.widgets import IPythonView


class ExampleIPythonViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_view_attached(view, gui):
            # This is called whenever a new view is added to the GUI.
            if isinstance(view, IPythonView):
                # We inject the first WaveformView of the GUI to the IPython console.
                view.inject(wv=gui.get_view(WaveformView))

        # Open an IPython view if there is not already one.
        controller.at_least_one_view('IPythonView')

```


## Customizing the feature view

In this example, we show how to customize the subplots in the feature view.

![image](https://user-images.githubusercontent.com/1942359/59465531-6625c500-8e2b-11e9-8442-c00530878959.png)

```python
# import from plugins/feature_view_custom_grid.py
"""Show how to customize the subplot grid specifiction in the feature view."""

import re
from phy import IPlugin, connect
from phy.cluster.views import FeatureView


def my_grid():
    """In the grid specification, 0 corresponds to the best channel, 1
    to the second best, and so on. A, B, C refer to the PC components."""
    s = """
    0A,1A 1A,2A 2A,0A
    0B,1B 1B,2B 2B,0B
    0C,1C 1C,2C 2C,0C
    """.strip()
    return [[_ for _ in re.split(' +', line.strip())] for line in s.splitlines()]


class ExampleCustomFeatureViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_view_attached(view, gui):
            if isinstance(view, FeatureView):
                # We change the specification of the subplots here.
                view.set_grid_dim(my_grid())

```


## Adding a custom color scheme to a view

In this example, we show how to add a custom color scheme to a view by showing clusters with a color that depends on the template amplitude.

```python
# import from plugins/color_scheme.py
"""Show how to add a custom color scheme to a view."""

from phy import IPlugin, connect
from phy.cluster.views import ClusterScatterView


class ExampleColorSchemePlugin(IPlugin):
    def attach_to_controller(self, controller):
        # Initial actions when creating views.
        @connect
        def on_view_attached(view, gui):
            # We need the initial list of cluster ids to initialize the color map.
            cluster_ids = controller.supervisor.clustering.cluster_ids

            if isinstance(view, ClusterScatterView):
                # Each view has a set of color schemes among which one can cycle through in
                # the GUI.
                view.add_color_scheme(
                    name='mycolorscheme',
                    fun=controller.get_cluster_amplitude,  # cluster_id => value
                    colormap='rainbow',  # or use a colorcet color map or a custom N*3 array
                    cluster_ids=cluster_ids,
                )

```


## Writing a custom scatter plot view

It is easy to add a new view that just shows a scatter plot, with one point per spike of the selected clusters, and custom 2D coordinates.

In this example, we show how to display a dimension reduction of the spike waveforms using the UMAP algorithm.

*Note*: this example requires the umap package, to install with `pip install umap-learn`

![image](https://user-images.githubusercontent.com/1942359/60545479-77bc0780-9d1b-11e9-91f0-6a491d2b82f8.png)

```python
# import from plugins/umap_view.py
"""Show how to write a custom dimension reduction view."""

from phy import IPlugin, Bunch
from phy.cluster.views import ScatterView


def umap(x):
    """Perform the dimension reduction of the array x."""
    from umap import UMAP
    return UMAP().fit_transform(x)


class WaveformUMAPView(ScatterView):
    """Every view corresponds to a unique view class, so we need to subclass ScatterView."""
    pass


class ExampleWaveformUMAPPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def coords(cluster_ids):
            """Must return a Bunch object with pos, spike_ids, spike_clusters."""
            # We select 200 spikes from the selected clusters.
            # WARNING: lasso and split will work but will *only split the shown subselection* of
            # spikes. You should use the `load_all` keyword argument to `coords()` to load all
            # spikes before computing the spikes inside the lasso, however (1) this could be
            # prohibitely long with UMAP, and (2) the coordinates will change when reperforming
            # the dimension reduction on all spikes, so the splitting would be meaningless anyway.
            # A warning is displayed when trying to split on a view that does not accept the
            # `load_all` keyword argument, because it means that all relevant spikes (even not
            # shown ones) are not going to be split.
            spike_ids = controller.selector(200, cluster_ids)
            # We get the cluster ids corresponding to the chosen spikes.
            spike_clusters = controller.supervisor.clustering.spike_clusters[spike_ids]
            # We get the waveforms of the spikes, across all channels so that we use the
            # same dimensions for every cluster.
            data = controller.model.get_waveforms(spike_ids, None)
            # We reshape the array as a 2D array so that we can pass it to the t-SNE algorithm.
            (n_spikes, n_samples, n_channels) = data.shape
            data = data.transpose((0, 2, 1))  # get an (n_spikes, n_channels, n_samples) array
            data = data.reshape((n_spikes, n_samples * n_channels))
            # We perform the dimension reduction.
            pos = umap(data)
            return Bunch(pos=pos, spike_ids=spike_ids, spike_clusters=spike_clusters)

        def create_view():
            """Create and return a histogram view."""
            return WaveformUMAPView(coords=controller.context.cache(coords))

        # Maps a view name to a function that returns a view
        # when called with no argument.
        controller.view_creator['WaveformUMAPView'] = create_view

```


## Adding a custom raw data filter

You can add one or several filters for the trace view and the waveform view. The shortcut `Alt+R` switches the different registered filters.

In this example, we add a Butterworth high-pass filter.

```python
# import from plugins/raw_data_filter.py
"""Show how to add a custom raw data filter for the TraceView and Waveform View

Use Alt+R in the GUI to toggle the filter.

"""

from scipy.signal import butter, filtfilt

from phy import IPlugin


class ExampleRawDataFilterPlugin(IPlugin):
    def attach_to_controller(self, controller):
        b, a = butter(3, 150.0 / controller.model.sample_rate * 2.0, 'high')

        @controller.raw_data_filter.add_filter
        def high_pass(arr, axis=0):
            return filtfilt(b, a, arr, axis=axis)

```


## Writing a custom matplotlib view

Most built-in views in phy are based on OpenGL instead of matplotlib, for performance reasons. Since writing OpenGL views is significantly more complex than with matplotlib, we cover OpenGL views later in this documentation.

In this example, we show how to create a custom view based on matplotlib. Specifically, we show a 2D histogram of spike features.

*Note*: once the plugin is activated, you need to explicitly open a new view with `View > New view > FeatureDensityView`.

![image](https://user-images.githubusercontent.com/1942359/58970988-ccbb3b00-87ba-11e9-93e9-c900d76a54f2.png)

```python
# import from plugins/matplotlib_view.py
"""Show how to create a custom matplotlib view in the GUI."""

from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas


class FeatureDensityView(ManualClusteringView):
    plot_canvas_class = PlotCanvasMpl  # use matplotlib instead of OpenGL (the default)

    def __init__(self, features=None):
        """features is a function (cluster_id => Bunch(data, ...)) where data is a 3D array."""
        super(FeatureDensityView, self).__init__()
        self.features = features

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return

        # To simplify, we only consider the first PC component of the first 2 best channels.
        # Note that the features are in sparse format, where data's shape is
        # (n_spikes, n_best_channels, n_pcs). Only best channels for that clusters are
        # considered.
        # For this example, we just take the first 2 dimensions.
        x, y = self.features(cluster_ids[0]).data[:, :2, 0].T

        # We draw a 2D histogram with matplotlib.
        # The objects are:
        # - self.figure, a Figure instance
        # - self.canvas, a PlotCanvasMpl instance
        # - self.canvas.ax, an Axes object.
        self.canvas.ax.hist2d(x, y, 50)

        # Use this to update the matplotlib figure.
        self.canvas.update()


class ExampleMatplotlibViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_feature_density_view():
            """A function that creates and returns a view."""
            return FeatureDensityView(features=controller._get_features)

        controller.view_creator['FeatureDensityView'] = create_feature_density_view

```


## Writing a custom OpenGL view

For increased performance, all built-in views in phy are not based on matplotlib, but on OpenGL. OpenGL is a real-time graphics programming interface developed for video games. It also provides hardware acceleration for fast display of large amounts of data.

In phy, OpenGL views are written on top of a thin layer, a fork of `glumpy.gloo` (object-oriented interface to OpenGL). On top of that, the `phy.plot` module proposes a minimal plotting API. This interface is complex as it suffers from the limitations of OpenGL. As such, writing custom OpenGL views for phy is not straightforward.

Here, we give a minimal example of a plugin implementing a custom OpenGL view. There is no in-depth documentation at the moment.

### Example

In this example, we simply display the template waveform on the peak channel of selected clusters.

*Note*: OpenGL is most useful when there is a lot of data (between tens of thousands and millions of points). For a plot as simple as this, you could as well use matplotlib. However, the method presented here scales well to plots with many more points.

![image](https://user-images.githubusercontent.com/1942359/59127305-9a9b0c00-8967-11e9-9dea-468a13fc98bc.png)


### Code

```python
# import from plugins/opengl_view.py
"""Show how to write a custom OpenGL view. This is for advanced users only."""

import numpy as np

from phy.utils.color import selected_cluster_color

from phy import IPlugin
from phy.cluster.views import ManualClusteringView
from phy.plot.visuals import PlotVisual


class MyOpenGLView(ManualClusteringView):
    """All OpenGL views derive from ManualClusteringView."""

    def __init__(self, templates=None):
        """
        Typically, the constructor takes as arguments *functions* that take as input
        one or several cluster ids, and return as many Bunch instances which contain
        the data as NumPy arrays. Many such functions are defined in the TemplateController.
        """

        super(MyOpenGLView, self).__init__()

        """
        The View instance contains a special `canvas` object which is a `̀PlotCanvas` instance.
        This class derives from `BaseCanvas` which itself derives from the PyQt5 `QOpenGLWindow`.
        The canvas represents a rectangular black window where you can draw geometric objects
        with OpenGL.

        phy uses the notion of **Layout** that lets you organize graphical elements in different
        subplots. These subplots can be organized in several ways:

        * Grid layout: a `(n_rows, n_cols)` grid of subplots (example: FeatureView).
        * Boxed layout: boxes arbitrarily located (example: WaveformView, using the
          probe geometry)
        * Stacked layout: one column with `n_boxes` subplots (example: TraceView,
          one row per channel)

        In this example, we use the stacked layout, with one subplot per cluster. This number
        will change at each cluster selection, depending on the number of selected clusters.
        But initially, we just use 1 subplot.

        """
        self.canvas.set_layout('stacked', n_plots=1)

        self.templates = templates

        """
        phy uses the notion of **Visual**. This is a graphical element that is represented with
        a single type of graphical element. phy provides many visuals:

        * PlotVisual (plots)
        * ScatterVisual (points with a given marker type and different colors and sizes)
        * LineVisual (for lines segments)
        * HistogramVisual
        * PolygonVisual
        * TextVisual
        * ImageVisual

        Each visual comes with a single OpenGL program, which is defined by a vertex shader
        and a fragment shader. These are programs written in a C-like language called GLSL.
        A visual also comes with a primitive type, which can be points, line segments, or
        triangles. This is all a GPU is able to render, but the position and the color of
        these primitives can be entirely customized in the shaders.

        The vertex shader acts on data arrays represented as NumPy arrays.

        These low-level details are hidden by the visuals abstraction, so it is unlikely that
        you'll ever need to write your own visual.

        In ManualClusteringViews, you typically define one or several visuals. For example
        if you need to add text, you would add `self.text_visual = TextVisual()`.

        """
        self.visual = PlotVisual()

        """
        For internal reasons, you need to add all visuals (empty for now) directly to the
        canvas, in the view's constructor. Later, we will use the `visual.set_data()` method
        to update the visual's data and display something in the figure.

        """
        self.canvas.add_visual(self.visual)

    def on_select(self, cluster_ids=(), **kwargs):
        """
        The main method to implement in ManualClusteringView is `on_select()`, called whenever
        new clusters are selected.

        *Note*: `cluster_ids` contains the clusters selected in the cluster view, followed
        by clusters selected in the similarity view.

        """

        """
        This method should always start with these few lines of code.
        """
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return

        """
        We update the number of boxes in the stacked layout, which is the number of
        selected clusters.
        """
        self.canvas.stacked.n_boxes = len(cluster_ids)

        """
        We obtain the template data.
        """
        bunchs = {cluster_id: self.templates(cluster_id).data for cluster_id in cluster_ids}

        """
        For performance reasons, it is best to use as few visuals as possible. In this example,
        we want 1 waveform template per subplot. We will use a single visual covering all
        subplots at once. This is the key to achieve good performance with OpenGL in Python.
        However, this comes with the drawback that the programming interface is more complicated.

        In principle, we would have to concatenate all data (x and y coordinates) of all subplots
        to pass it to `self.visual.set_data()` in order to draw all subplots at once. But this
        is tedious.

        phy uses the notion of **batch**: for each subplot, we set *partial data* for the subplot
        which just prepares the data for concatenation *after* we're done with looping through
        all clusters. The concatenation happens in the special call
        `self.canvas.update_visual(self.visual)`.

        We need to call `visual.reset_batch()` before constructing a batch.

        """
        self.visual.reset_batch()

        """
        We iterate through all selected clusters.
        """
        for idx, cluster_id in enumerate(cluster_ids):
            bunch = bunchs[cluster_id]

            """
            In this example, we just keep the peak channel. Note that `bunch.template` is a
            2D array `(n_samples, n_channels)` where `n_channels` in the number of "best"
            channels for the cluster. The channels are sorted by decreasing template amplitude,
            so the first one is the peak channel. The channel ids can be found in
            `bunch.channel_ids`.
            """
            y = bunch.template[:, 0]

            """
            We decide to use, on the x axis, values ranging from -1 to 1. This is the
            standard viewport in OpenGL and phy.
            """
            x = np.linspace(-1., 1., len(y))

            """
            phy requires you to specify explicitly the x and y range of the plots.
            The `data_bounds` variable is a `(xmin, ymin, xmax, ymax)` tuple representing the
            lower-left and upper-right corners of a rectangle. By default, the data bounds
            of the entire view is (-1, -1, 1, 1), also called normalized device coordinates.
            Eventually, OpenGL uses this coordinate system for display, but phy provides
            a transform system to convert from different coordinate systems, both on the CPU
            and the GPU.

            Here, the x range is (-1, 1), and the y range is (m, M) where m and M are
            respectively the min and max of the template.
            """
            m, M = y.min(), y.max()
            data_bounds = (-1, m, +1, M)

            """
            This function gives the color of the i-th selected cluster. This is a 4-tuple with
            values between 0 and 1 for RGBA: red, green, blue, alpha channel (transparency,
            1 by default).
            """
            color = selected_cluster_color(idx)

            """
            The plot visual takes as input the x and y coordinates of the points, the color,
            and the data bounds.
            There is also a special keyword argument `box_index` which is the subplot index.
            In the stacked layout, this is just an integer identifying the subplot index, from
            top to bottom. Note that in the grid view, the box index is a pair (row, col).
            """
            self.visual.add_batch_data(
                x=x, y=y, color=color, data_bounds=data_bounds, box_index=idx)

        """
        After the loop, this special call automatically builds the data to upload to the GPU
        by concatenating the partial data set in `add_batch_data()`.
        """
        self.canvas.update_visual(self.visual)

        """
        After updating the data on the GPU, we need to refresh the canvas.
        """
        self.canvas.update()


class ExampleOpenGLViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_my_view():
            return MyOpenGLView(templates=controller._get_template_waveforms)

        controller.view_creator['MyOpenGLView'] = create_my_view

        # Open a view if there is not already one.
        controller.at_least_one_view('MyOpenGLView')

```
