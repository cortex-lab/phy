# Customization

The **phy user directory** is `~/.phy/`, where `~` is your home directory. On Linux, it is typically `/home/username/.phy/`.

The **phy user config file** is `~/.phy/phy_config.py`. It is a Python file. A default file is automatically created if needed.

## Writing plugins for phy

You can extend the GUI by writing **plugins**. This involves the following steps:

1. Make sure the plugin code is stored in a file that is loaded by phy.
2. Write a plugin by creating a class deriving from `phy.IPlugin`
3. Add the plugin name to the list of plugins loaded by the GUI.


### 1. Where to write the code

You can write the code either:

* In your phy user config file `~/.phy/phy_config.py`
* Or in a separate Python file saved in `~/.phy/plugins/myplugin.py`

*Warning*: make sure that the same plugin (using the same name) is not defined in two different places, otherwise it might not be loaded properly.

### 2. Writing the plugin

Here is an example of a "hello world" plugin:

```python
from phy import IPlugin

class MyPlugin(IPlugin):
    def attach_to_controller(self, controller):
        print("Hello world!")
```

Put this code either in `~/.phy/phy_config.py` or in `~/.phy/plugins/myplugin.py`. If the latter, remove the possibly existing `MyPlugin` class definition in `~/.phy/phy_config.py`.

When a plugin is activated, the `attach_to_controller(controller)` function is automatically called with the `TemplateController` instance as argument. This object gives you programmatic access to all objects, data, and views in the GUI.

### 3. Activating the plugin

Put this line in your `~/.phy/phy_config.py`:

```python
c.TemplateGUI.plugins = ['MyPlugin']
```

Then, launching the GUI should print `Hello world!`


## Using the event system

A lightweight **event system** is implemented in phy. You will likely have to use it when writing plugins, so here is a brief overview.

Objects in the program may emit **events**. An event has a name and optional parameters. Other objects may subscribe to events by **registering callbacks**. A callback is a function that is called when an event is raised.


### Registering a callback

In this example, we register a callback function to the event `myevent`:

```python
from phy import connect

@connect
def on_myevent(sender, arg):
    """The function name pattern is on_eventname()."""
    print("myevent called with argument %s" % arg)
```

*Note*: the first argument, `sender`, is mandatory. You can use arbitrary further positional and keyword arguments.


### Emitting an event

In this example, the object `sender` (any Python object) emits a `myevent` event with argument `123`:

```python
from phy import emit
emit('myevent', sender, 123)
```

This will automatically call all callbacks registered to `myevent`, in the order they were registered. So here, it will display:

```
myevent called with argument 123
```


### Optional keyword arguments to `connect()`

The `connect()` function has a few optional parameters.

* `connect(f, event=event)` to specify the event name explicitely, without having to use a special `on_eventname()` name for the function.
* `connect(f, sender=sender)` to restrict the callback to a specific sender.

*Note*: events are sent globally in the Python process.

You can also filter senders directly in the callback function. For example, here is how to write a callback function that will only react to events sent by instances of the `AcceptableSender` class.

```
@connect
def on_myevent(sender, arg):
    if not isinstance(sender, AcceptableSender):
        return
```


## Main objects in phy

The main objects used in phy are the following.

### TemplateController

This is the main object, that holds together the model (access to the data), the views, and the manual clustering logic.

### TemplateModel

This object (`controller.model`) holds the data: spike times, template waveforms, raw data, initial template assignments, etc.

*Note*: this class is defined in `phylib.io.model`.

### Supervisor

This object (`controller.supervisor`) is responsible for creating the cluster and similarity views, defining all clustering actions, specifying the logic of the wizard, etc. It holds together more specialized classes that you should not have to know about:

* **Clustering**: manages the cluster assignments and the related undo stack.
* **ClusterMeta**: manages the cluster groups and labels, and the related undo stack.
* **History**: a generic undo stack used by the two classes above.
* **HTMLWidget**: a generic HTML widget with Javascript-Python communication handled by PyQt5.
* **Table**: a table used by the cluster and similarity views.
* **Context**: manages the memory and disk cache, using joblib.
* **ClusterColorSelector**: manages the cluster color mapping.

*Note*: the code of the table is in a separate Javascript project, `tablejs` that uses the `ListJS` library.


### GUI

This object provides a lightweight generic GUI with the following features:

* Dock widgets for the views (HTML, OpenGL, matplotlib)
* Status bar
* Menu bar and submenus
* Keyboard shortcuts
* Snippets
* GUI state (saving GUI- and view-specific options globally and locally)


## Events used in phy

In your plugins, you can register custom functions to existing events that are used in phy. This is how you can extend the GUI.

Important events used in phy include the following (ordered by the different objects that can send the events).

Remember the generic form of event emission: `emit(event_name, sender, argument)`.


### Controller

* `emit('gui_ready', controller, gui)`: when the GUI has been fully loaded.

### Cluster view

* `emit('table_filter', cluster_view, cluster_ids)`: when cluster filtering changes in the cluster view.
* `emit('table_sort', cluster_view, cluster_ids)`: when the cluster order changes in the cluster view.

### Supervisor

* `emit('attach_gui', supervisor)`: when the Supervisor has been attached to the GUI.
* `emit('select', supervisor, cluster_ids)`: when the cluster selection changes.
* `emit('cluster', supervisor, up)`: when cluster assignments, cluster groups, or cluster labels change. `up` is an `UpdateInfo` object that contains all the information about the cluster changes.
* `emit('color_mapping_changed', supervisor)`: when a different color mapping is selected.

#### UpdateInfo instance

This class derives from `phy.Bunch`, which itself derives from the Python built-in `dict` type. It is a dictionary with an extra `a.b` syntax that is more convenient than `a["b"]`.

Main `UpdateInfo` attributes include:

* `description`: information about the update: `merge`, `assign` (for split), or `metadata_name` for group/label changes
* `history`: `None`, `undo`, or `redo`
* `added`: list of new clusters
* `deleted`: list of old clusters (remember that cluster ids do not change, there can be only be added clusters or deleted clusters)
* `spike_ids`: all spike ids affected by the update
* `descendants`: list of pairs `(old_cluster_id, new_cluster_id)`, used to track the history of clusters
* `metadata_changed`: clusters that had changed metadata (group or label)
* `metadata_value`: new value of the metadata (group or label)


### View

The following events are raised with **Control+click** in specific views:

* `emit('cluster_click', view, cluster_id, button=None)` when a cluster is selected in the raster or template view.
* `emit('spike_click', trace_view, channel_id=None, spike_id=None, cluster_id=None)`: when a spike is selected in the trace view.
* `emit('channel_click', waveform_view, channel_id=None, key=None, button=None)`: when a channel is selected in the waveform view.

### GUI

* `emit('add_view', gui, view)`: when a view is added to the GUI.
* `emit('close_view', gui, view)`: when a view is closed in the GUI.
* `emit('show', gui)`: when the GUI is shown.
* `emit('close', gui)`: when the GUI is closed.


## Plugin examples

In this section, we give a few examples of plugins/

### Hello world

Here is how to write a useless plugin that displays a message every time there's a new cluster assignment:

```python
from phy import IPlugin, connect

class MyPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect(sender=controller)
        def on_gui_ready(gui):
            """This is called when the GUI and all objects are fully loaded.
            This is to make sure that controller.supervisor is properly defined.
            """

            @connect(sender=controller.supervisor)
            def on_cluster(sender, up):
                """This is called every time a cluster assignment or cluster group/label changes."""
                print("Clusters update: %s" % up)
```

This displays the following in the console, for example:

```
Clusters update: <merge [212, 299] => [305]>
Clusters update: <metadata_group [305] => good>
Clusters update: <metadata_neurontype [81, 82] => interneuron>
```


### Changing the number of spikes in views

You can customize the maximum number of spikes in the different views as follows (the example below shows the default values):

```python
from phy import IPlugin, connect

class MyPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Feel free to keep below just the values you need to change."""
        controller.n_spikes_waveforms = 100
        controller.batch_size_waveforms = 10
        controller.n_spikes_features = 2500
        controller.n_spikes_features_background = 1000
        controller.n_spikes_amplitudes = 5000
        controller.n_spikes_correlograms = 100000
```

*Note*: you need to manually delete the `.phy` subdirectory within your data directory when changing these parameters, otherwise errors will happen in the GUI.


### Defining a custom cluster metrics

In addition to cluster labels that you can create and modify in the cluster view, you can also define **cluster metrics**.

A cluster metrics is a function that assigns a scalar to every cluster. For example, any algorithm computing a kind of cluster quality is a cluster metrics.

You can define your own cluster metrics in a plugin. The values are then shown in a new column in the cluster view, and recomputed automatically when changing cluster assignments.

You can use the controller's methods to access the data. In rare cases, you may need to access the model directly (via `controller.model`).

For example, here is a custom cluster metrics that shows the mean inter-spike interval.

```python
import numpy as np
from phy import IPlugin, connect

class MyPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Note that this function is called at initialization time, *before* the supervisor is
        created. The `controller.cluster_metrics` items are then passed to the supervisor when
        constructing it."""

        def meanisi(cluster_id):
            t = controller.get_spike_times(cluster_id).data
            return np.diff(t).mean()

        # Use this dictionary to define custom cluster metrics.
        controller.cluster_metrics['meanisi'] = meanisi
```


### Writing a custom cluster statistics view

The ISI view, firing rate view, and amplitude histogram view all share the same characteristics. These views show histograms of cluster-dependent values. The number of bins and the maximum bin can be customized. All of these views derive from the **HistogramView**, a generic class for cluster statistics that can be displayed as histograms.

In the following example, we define a custom cluster statistics views using the PC features.

```python
from phy import IPlugin, Bunch
from phy.cluster.views import HistogramView


class FeatureHistogramView(HistogramView):
    n_bins = 100  # default number of bins
    x_max = .1  # maximum value on the x axis (maximum bin)
    alias_char = 'fh'  # provide `fhn` (set number of bins) and `fhm` (set max bin) snippets


class MyPlugin(IPlugin):
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


### Writing a custom matplotlib view

Most built-in views in phy are based on OpenGL instead of matplotlib, for performance reasons. Since writing OpenGL views is significantly more complex than with matplotlib, we cover OpenGL views in the developer section later in this documentation.

In this example, we show how to create a custom view based on matplotlib. Specifically, we show a 2D histogram of spike features.

![image](https://user-images.githubusercontent.com/1942359/58970988-ccbb3b00-87ba-11e9-93e9-c900d76a54f2.png)

```python
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


class MyPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_feature_density_view():
            """A function that creates and returns a view."""
            return FeatureDensityView(features=controller.get_features)

        controller.view_creator['FeatureDensityView'] = create_feature_density_view

```
