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

Events are raised globally. Every object in the Python process may subscribe a callback function to any event.


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
    # Now, process the event as we're sure the sender derives from the AcceptableSender class.
```


## Main objects in phy

The main objects used in phy are the following.

### TemplateController

This is the main object, that holds together the model (access to the data), the views, and the manual clustering logic. It defines a method to create and display the GUI.

### TemplateModel

This object (`controller.model`) holds the data: spike times, template waveforms, raw data, initial template assignments, etc.

A few remarks:

* There is a distinction between **templates** and **clusters**: spike templates assignments are done by the spike sorting algorithm (e.g. KiloSort), whereas spike clusters assignments are manual refinements of the original spike templates assignments. Therefore:
    * `spike_templates.npy` is created by the spike sorting algorithm, and never modified by phy
    * `spike_clusters.npy` is initially a copy of `spike_templates.npy`, but it is modified by phy during the manual clustering session
* The TemplateModel supports both dense and sparse arrays for templates, features...
* In several methods, the TemplateModel requires the channel ids to be passed explicitly. There are methods to find the "best" channels associated to a given template or cluster.


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

#### Context

Disk cache and memory cache are stored in the `.phy` subdirectory within the data directory. Functions retrieving cluster-dependent data such as waveforms, templates, and so on, are all cached for performance reasons. It is important to ensure that this directory is stored on an SSD.


### GUI

This object provides a lightweight generic GUI with the following features:

* Dock widgets for the views (HTML, OpenGL, matplotlib)
* Status bar
* Actions
* Menu bar and submenus
* Keyboard shortcuts
* Snippets
* GUI state

#### Actions

Default menus are accessible with `gui.file_actions`, `gui.select_actions`, and so on. These are instances of the `Actions` class. The main method of `Actions` is `Actions.add()`. This method has many arguments, see the Plugins section for a few examples, and the API documentation for more details.


#### GUI state

The **GUI state** contains information about the GUI, like the position, size, dock widget layout, and view options that can be changed via the menu or keyboard shortcuts, like the number of bins in the correlogram view and so on. The GUI state is saved in two JSON files:

* Global state: in `~/.phy/TemplateGUI/state.json`
* Local state: in `<data_directory>/.phy/state.json`

Values saved in the local state (like data-dependent scaling in the views) override those saved in the global state.

These files are automatically recreated if they're missing.

*Note*: in case of visualization problem in the GUI, feel free to delete these files.


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

This is an object that stores information relevant to a clustering action (merge, split, move, label, undo, redo...). It is passed as an argument to the `cluster(up)` event.

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


## Examples of using the API

You can use the phy API to write custom analysis and visualization scripts, without using the GUI. We give a few examples here.

*Note*: the high-level methods provided by `TemplateModel` are convenient but not particularly efficient. If you need better performance, you should use the `TemplateController` class which leverages the cache in the `.phy` subdirectory.

### Extracting waveforms

Here is a Python script that will extract the waveforms of a given cluster directly from the raw data, and display them.

![image](https://user-images.githubusercontent.com/1942359/59498589-fc8ed080-8e95-11e9-86b6-b251f8c887be.png)

```python
import sys
import matplotlib.pyplot as plt
from phylib.io.model import load_model
from phylib.utils.color import selected_cluster_color

# First, we load the TemplateModel.
model = load_model(sys.argv[1])  # first argument: path to params.py

# We obtain the cluster id from the command-line arguments.
cluster_id = int(sys.argv[2])  # second argument: cluster index

# We get the waveforms of the cluster.
waveforms = model.get_cluster_spike_waveforms(cluster_id)
n_spikes, n_samples, n_channels_loc = waveforms.shape

# We get the channel ids where the waveforms are located.
channel_ids = model.get_cluster_channels(cluster_id)

# We plot the waveforms on the first four channels.
f, axes = plt.subplots(1, min(4, n_channels_loc), sharey=True)
for ch in range(min(4, n_channels_loc)):
    axes[ch].plot(waveforms[::100, :, ch].T, c=selected_cluster_color(0, .05))
    axes[ch].set_title("channel %d" % channel_ids[ch])
plt.show()
```

### Launching the Template GUI

Simply use the following script to launch the GUI from a Python script.

```python
from phy.apps.template import template_gui

# Launch the GUI using the params.py file found in the current directory.
template_gui('params.py')
```
