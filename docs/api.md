# API documentation of phy

phy: interactive visualization and manual spike sorting of large-scale ephys data.

## Table of contents

### [phy.utils](#phyutils)

* [phy.utils.add_alpha](#phyutilsadd_alpha)
* [phy.utils.attach_plugins](#phyutilsattach_plugins)
* [phy.utils.ensure_dir_exists](#phyutilsensure_dir_exists)
* [phy.utils.load_json](#phyutilsload_json)
* [phy.utils.load_master_config](#phyutilsload_master_config)
* [phy.utils.load_pickle](#phyutilsload_pickle)
* [phy.utils.phy_config_dir](#phyutilsphy_config_dir)
* [phy.utils.read_python](#phyutilsread_python)
* [phy.utils.read_text](#phyutilsread_text)
* [phy.utils.read_tsv](#phyutilsread_tsv)
* [phy.utils.save_json](#phyutilssave_json)
* [phy.utils.save_pickle](#phyutilssave_pickle)
* [phy.utils.selected_cluster_color](#phyutilsselected_cluster_color)
* [phy.utils.write_text](#phyutilswrite_text)
* [phy.utils.write_tsv](#phyutilswrite_tsv)
* [phy.utils.Bunch](#phyutilsbunch)
* [phy.utils.ClusterColorSelector](#phyutilsclustercolorselector)
* [phy.utils.Context](#phyutilscontext)
* [phy.utils.IPlugin](#phyutilsiplugin)


### [phy.gui](#phygui)

* [phy.gui.busy_cursor](#phyguibusy_cursor)
* [phy.gui.create_app](#phyguicreate_app)
* [phy.gui.input_dialog](#phyguiinput_dialog)
* [phy.gui.is_high_dpi](#phyguiis_high_dpi)
* [phy.gui.message_box](#phyguimessage_box)
* [phy.gui.prompt](#phyguiprompt)
* [phy.gui.require_qt](#phyguirequire_qt)
* [phy.gui.run_app](#phyguirun_app)
* [phy.gui.screen_size](#phyguiscreen_size)
* [phy.gui.screenshot](#phyguiscreenshot)
* [phy.gui.thread_pool](#phyguithread_pool)
* [phy.gui.Actions](#phyguiactions)
* [phy.gui.Debouncer](#phyguidebouncer)
* [phy.gui.DockWidget](#phyguidockwidget)
* [phy.gui.GUI](#phyguigui)
* [phy.gui.GUIState](#phyguiguistate)
* [phy.gui.HTMLBuilder](#phyguihtmlbuilder)
* [phy.gui.HTMLWidget](#phyguihtmlwidget)
* [phy.gui.IPythonView](#phyguiipythonview)
* [phy.gui.KeyValueWidget](#phyguikeyvaluewidget)
* [phy.gui.Snippets](#phyguisnippets)
* [phy.gui.Table](#phyguitable)
* [phy.gui.Worker](#phyguiworker)


### [phy.plot](#phyplot)

* [phy.plot.Subplot](#phyplotsubplot)
* [phy.plot.extend_bounds](#phyplotextend_bounds)
* [phy.plot.get_linear_x](#phyplotget_linear_x)
* [phy.plot.Axes](#phyplotaxes)
* [phy.plot.AxisLocator](#phyplotaxislocator)
* [phy.plot.BaseCanvas](#phyplotbasecanvas)
* [phy.plot.BaseLayout](#phyplotbaselayout)
* [phy.plot.BaseVisual](#phyplotbasevisual)
* [phy.plot.BatchAccumulator](#phyplotbatchaccumulator)
* [phy.plot.Boxed](#phyplotboxed)
* [phy.plot.GLSLInserter](#phyplotglslinserter)
* [phy.plot.Grid](#phyplotgrid)
* [phy.plot.HistogramVisual](#phyplothistogramvisual)
* [phy.plot.ImageVisual](#phyplotimagevisual)
* [phy.plot.Lasso](#phyplotlasso)
* [phy.plot.LineVisual](#phyplotlinevisual)
* [phy.plot.PanZoom](#phyplotpanzoom)
* [phy.plot.PlotCanvas](#phyplotplotcanvas)
* [phy.plot.PlotVisual](#phyplotplotvisual)
* [phy.plot.PolygonVisual](#phyplotpolygonvisual)
* [phy.plot.Range](#phyplotrange)
* [phy.plot.Scale](#phyplotscale)
* [phy.plot.ScatterVisual](#phyplotscattervisual)
* [phy.plot.TextVisual](#phyplottextvisual)
* [phy.plot.TransformChain](#phyplottransformchain)
* [phy.plot.Translate](#phyplottranslate)
* [phy.plot.UniformPlotVisual](#phyplotuniformplotvisual)
* [phy.plot.UniformScatterVisual](#phyplotuniformscattervisual)


### [phy.cluster](#phycluster)

* [phy.cluster.select_traces](#phyclusterselect_traces)
* [phy.cluster.AmplitudeView](#phyclusteramplitudeview)
* [phy.cluster.ClusterMeta](#phyclusterclustermeta)
* [phy.cluster.ClusterScatterView](#phyclusterclusterscatterview)
* [phy.cluster.ClusterView](#phyclusterclusterview)
* [phy.cluster.Clustering](#phyclusterclustering)
* [phy.cluster.CorrelogramView](#phyclustercorrelogramview)
* [phy.cluster.FeatureView](#phyclusterfeatureview)
* [phy.cluster.FiringRateView](#phyclusterfiringrateview)
* [phy.cluster.HistogramView](#phyclusterhistogramview)
* [phy.cluster.ISIView](#phyclusterisiview)
* [phy.cluster.ManualClusteringView](#phyclustermanualclusteringview)
* [phy.cluster.ProbeView](#phyclusterprobeview)
* [phy.cluster.RasterView](#phyclusterrasterview)
* [phy.cluster.ScatterView](#phyclusterscatterview)
* [phy.cluster.SimilarityView](#phyclustersimilarityview)
* [phy.cluster.Supervisor](#phyclustersupervisor)
* [phy.cluster.TemplateView](#phyclustertemplateview)
* [phy.cluster.TraceImageView](#phyclustertraceimageview)
* [phy.cluster.TraceView](#phyclustertraceview)
* [phy.cluster.UpdateInfo](#phyclusterupdateinfo)
* [phy.cluster.WaveformView](#phyclusterwaveformview)


### [phy.apps](#phyapps)

* [phy.apps.add_default_handler](#phyappsadd_default_handler)
* [phy.apps.capture_exceptions](#phyappscapture_exceptions)
* [phy.apps.contextmanager](#phyappscontextmanager)
* [phy.apps.exceptionHandler](#phyappsexceptionhandler)
* [phy.apps.format_exception](#phyappsformat_exception)
* [phy.apps.BaseController](#phyappsbasecontroller)
* [phy.apps.FeatureMixin](#phyappsfeaturemixin)
* [phy.apps.Path](#phyappspath)
* [phy.apps.QtDialogLogger](#phyappsqtdialoglogger)
* [phy.apps.TemplateMixin](#phyappstemplatemixin)
* [phy.apps.TraceMixin](#phyappstracemixin)
* [phy.apps.WaveformMixin](#phyappswaveformmixin)


### [phy.apps.template](#phyappstemplate)

* [phy.apps.template.from_sparse](#phyappstemplatefrom_sparse)
* [phy.apps.template.get_template_params](#phyappstemplateget_template_params)
* [phy.apps.template.load_model](#phyappstemplateload_model)
* [phy.apps.template.template_describe](#phyappstemplatetemplate_describe)
* [phy.apps.template.template_gui](#phyappstemplatetemplate_gui)
* [phy.apps.template.TemplateController](#phyappstemplatetemplatecontroller)
* [phy.apps.template.TemplateModel](#phyappstemplatetemplatemodel)


### [phy.apps.kwik](#phyappskwik)

* [phy.apps.kwik.kwik_describe](#phyappskwikkwik_describe)
* [phy.apps.kwik.kwik_gui](#phyappskwikkwik_gui)
* [phy.apps.kwik.KwikController](#phyappskwikkwikcontroller)




## phy.utils

Utilities: plugin system, event system, configuration system, profiling, debugging, cacheing,
basic read/write functions.

---

#### phy.utils.add_alpha


**`phy.utils.add_alpha(c, alpha=1.0)`**

Add an alpha channel to an RGB color.

**Parameters**


* `c : array-like (2D, shape[1] == 3) or 3-tuple`

* `alpha : float`

---

#### phy.utils.attach_plugins


**`phy.utils.attach_plugins(controller, plugins=None, config_dir=None, dirs=None)`**

Attach plugins to a controller object.

Attached plugins are those found in the user configuration file for the given gui_name or
class name of the Controller instance, plus those specified in the plugins keyword argument.

**Parameters**


* `controller : object`
    The controller object that will be passed to the `attach_to_controller()` plugins methods.

* `plugins : list of str`
    List of plugin names to attach in addition to those found in the user configuration file.

* `config_dir : str`
    Path to the user configuration file. By default, the directory is `~/.phy/`.

---

#### phy.utils.ensure_dir_exists


**`phy.utils.ensure_dir_exists(path)`**

Ensure a directory exists, and create it otherwise.

---

#### phy.utils.load_json


**`phy.utils.load_json(path)`**

Load a JSON file.

---

#### phy.utils.load_master_config


**`phy.utils.load_master_config(config_dir=None)`**

Load a master Config file from the user configuration file (by default, this is
`~/.phy/phy_config.py`).

---

#### phy.utils.load_pickle


**`phy.utils.load_pickle(path)`**

Load a pickle file using joblib.

---

#### phy.utils.phy_config_dir


**`phy.utils.phy_config_dir()`**

Return the absolute path to the phy user directory. By default, `~/.phy/`.

---

#### phy.utils.read_python


**`phy.utils.read_python(path)`**

Read a Python file.

**Parameters**


* `path : str or Path`

**Returns**


* `metadata : dict`
    A dictionary containing all variables defined in the Python file (with `exec()`).

---

#### phy.utils.read_text


**`phy.utils.read_text(path)`**

Read a text file.

---

#### phy.utils.read_tsv


**`phy.utils.read_tsv(path)`**

Read a CSV/TSV file.

**Returns**


* `data : list of dicts`

---

#### phy.utils.save_json


**`phy.utils.save_json(path, data)`**

Save a dictionary to a JSON file.

Support NumPy arrays and QByteArray objects. NumPy arrays are saved as base64-encoded strings,
except for 1D arrays with less than 10 elements, which are saved as a list for human
readability.

---

#### phy.utils.save_pickle


**`phy.utils.save_pickle(path, data)`**

Save data to a pickle file using joblib.

---

#### phy.utils.selected_cluster_color


**`phy.utils.selected_cluster_color(i, alpha=1.0)`**

Return the color, as a 4-tuple, of the i-th selected cluster.

---

#### phy.utils.write_text


**`phy.utils.write_text(path, contents)`**

Write a text file.

---

#### phy.utils.write_tsv


**`phy.utils.write_tsv(path, data, first_field=None, exclude_fields=(), n_significant_figures=4)`**

Write a CSV/TSV file.

**Parameters**


* `data : list of dicts`

* `first_field : str`
    The name of the field that should come first in the file.

* `exclude_fields : list-like`
    Fields present in the data that should not be saved in the file.

* `n_significant_figures : int`
    Number of significant figures used for floating-point numbers in the file.

---

### phy.utils.Bunch

A subclass of dictionary with an additional dot syntax.

---

#### Bunch.copy


**`Bunch.copy(self)`**

Return a new Bunch instance which is a copy of the current Bunch instance.

---

### phy.utils.ClusterColorSelector

Assign a color to clusters depending on cluster labels or metrics.

---

#### ClusterColorSelector.get


**`ClusterColorSelector.get(self, cluster_id, alpha=None)`**

Return the RGBA color of a single cluster.

---

#### ClusterColorSelector.get_colors


**`ClusterColorSelector.get_colors(self, cluster_ids, alpha=1.0)`**

Return the RGBA colors of some clusters.

---

#### ClusterColorSelector.get_values


**`ClusterColorSelector.get_values(self, cluster_ids)`**

Get the values of clusters for the selected color field..

---

#### ClusterColorSelector.map


**`ClusterColorSelector.map(self, values)`**

Convert values to colors using the selected colormap.

**Parameters**


* `values : array-like (1D)`

**Returns**


* `colors : array-like (2D, shape[1] == 3)`

---

#### ClusterColorSelector.set_cluster_ids


**`ClusterColorSelector.set_cluster_ids(self, cluster_ids)`**

Precompute the value range for all clusters.

---

#### ClusterColorSelector.set_color_mapping


**`ClusterColorSelector.set_color_mapping(self, fun=None, colormap=None, categorical=None, logarithmic=None)`**

Set the field used to choose the cluster colors, and the associated colormap.

**Parameters**


* `fun : function`
    Function cluster_id => value

* `colormap : array-like`
    A `(N, 3)` array with the colormaps colors

* `categorical : boolean`
    Whether the colormap is categorical (one value = one color) or continuous (values
    are continuously mapped from their initial interval to the colors).

* `logarithmic : boolean`
    Whether to use a logarithmic transform for the mapping.

---

### phy.utils.Context

Handle function disk and memory caching with joblib.

Memcaching a function is used to save *in memory* the output of the function for all
passed inputs. Input should be hashable. NumPy arrays are supported. The contents of the
memcache in memory can be persisted to disk with `context.save_memcache()` and
`context.load_memcache()`.

Caching a function is used to save *on disk* the output of the function for all passed
inputs. Input should be hashable. NumPy arrays are supported. This is to be preferred
over memcache when the inputs or outputs are large, and when the computations are longer
than loading the result from disk.

**Constructor**


* `cache_dir : str`
    The directory in which the cache will be created.

* `verbose : int`
    The verbosity level passed to joblib Memory.

**Examples**

```python
@context.memcache
def my_function(x):
    return x * x

@context.cache
def my_function(x):
    return x * x
```

---

#### Context.cache


**`Context.cache(self, f)`**

Cache a function using the context's cache directory.

---

#### Context.load


**`Context.load(self, name, location='local')`**

Load a dictionary saved in the cache directory.

**Parameters**


* `name : str`
    The name of the object to save to disk.

* `location : str`
    Can be `local` or `global`.

---

#### Context.load_memcache


**`Context.load_memcache(self, name)`**

Load the memcache from disk (pickle file), if it exists.

---

#### Context.memcache


**`Context.memcache(self, f)`**

Cache a function in memory using an internal dictionary.

---

#### Context.save


**`Context.save(self, name, data, location='local', kind='json')`**

Save a dictionary in a JSON/pickle file within the cache directory.

**Parameters**


* `name : str`
    The name of the object to save to disk.

* `data : dict`
    Any serializable dictionary that will be persisted to disk.

* `location : str`
    Can be `local` or `global`.

* `kind : str`
    Can be `json` or `pickle`.

---

#### Context.save_memcache


**`Context.save_memcache(self)`**

Save the memcache to disk using pickle.

---

### phy.utils.IPlugin

All plugin classes should derive from this class.

Plugin classes should just implement a method `attach_to_controller(self, controller)`.

---

## phy.gui

GUI routines.

---

#### phy.gui.busy_cursor


**`phy.gui.busy_cursor(activate=True)`**

Context manager displaying a busy cursor during a long command.

---

#### phy.gui.create_app


**`phy.gui.create_app()`**

Create a Qt application.

---

#### phy.gui.input_dialog


**`phy.gui.input_dialog(title, sentence, text=None)`**

Display a dialog with a text box.

**Parameters**


* `title : str`
    Title of the dialog.

* `sentence : str`
    Message of the dialog.

* `text : str`
    Default text in the text box.

---

#### phy.gui.is_high_dpi


**`phy.gui.is_high_dpi()`**

Return whether the screen has a high density.

Note: currently, this only returns whether the screen width is greater than an arbitrary
value chosen at 3000.

---

#### phy.gui.message_box


**`phy.gui.message_box(message, title='Message', level=None)`**

Display a message box.

**Parameters**

* `message : str`

* `title : str`

* `level : str`
    information, warning, or critical

---

#### phy.gui.prompt


**`phy.gui.prompt(message, buttons=('yes', 'no'), title='Question')`**

Display a dialog with several buttons to confirm or cancel an action.

**Parameters**


* `message : str`
    Dialog message.

* `buttons : tuple`
    Name of the standard buttons to show in the prompt: yes, no, ok, cancel, close, etc.
    See the full list at https://doc.qt.io/qt-5/qmessagebox.html#StandardButton-enum

* `title : str`
    Dialog title.

---

#### phy.gui.require_qt


**`phy.gui.require_qt(func)`**

Function decorator to specify that a function requires a Qt application.

Use this decorator to specify that a function needs a running
Qt application before it can run. An error is raised if that is not
the case.

---

#### phy.gui.run_app


**`phy.gui.run_app()`**

Run the Qt application.

---

#### phy.gui.screen_size


**`phy.gui.screen_size()`**

Return the screen size as a tuple (width, height).

---

#### phy.gui.screenshot


**`phy.gui.screenshot(widget, path=None, dir=None)`**

Save a screenshot of a Qt widget to a PNG file.

By default, the screenshots are saved in `~/.phy/screenshots/`.

**Parameters**


* `widget : Qt widget`
    Any widget to capture (including OpenGL widgets).

* `path : str or Path`
    Path to the PNG file.

---

#### phy.gui.thread_pool


**`phy.gui.thread_pool()`**

Return a QThreadPool instance that can `start()` Worker instances for multithreading.

**Example**

```python
w = Worker(print, "hello world")
thread_pool().start(w)
```

---

### phy.gui.Actions

Group of actions bound to a GUI.

This class attaches to a GUI and implements the following features:

* Add and remove actions
* Keyboard shortcuts for the actions
* Display all shortcuts

**Constructor**


* `gui : GUI instance`

* `name : str`
    Name of this group of actions.

* `menu : str`
    Name of the GUI menu that will contain the actions.

* `submenu : str`
    Name of the GUI submenu that will contain the actions.

* `default_shortcuts : dict`
    Map action names to keyboard shortcuts (regular strings).

* `default_snippets : dict`
    Map action names to snippets (regular strings).

---

#### Actions.add


**`Actions.add(self, callback=None, name=None, shortcut=None, alias=None, prompt=False, n_args=None, docstring=None, menu=None, submenu=None, view=None, view_submenu=None, verbose=True, checkable=False, checked=False, set_busy=False, prompt_default=None, show_shortcut=True, icon=None, toolbar=False)`**

Add an action with a keyboard shortcut.

**Parameters**


* `callback : function`
    Take no argument if checkable is False, or a boolean (checked) if it is True

* `name : str`
    Action name, the callback's name by default.

* `shortcut : str`
    The keyboard shortcut for this action.

* `alias : str`
    Snippet, the name by default.

* `prompt : boolean`
    Whether this action should display a dialog with an input box where the user can
    write arguments to the callback function.

* `n_args : int`
    If prompt is True, specify the number of expected arguments.

* `set_busy : boolean`
    Whether to use a busy cursor while performing the action.

* `prompt_default : str`
    The default text in the input text box, if prompt is True.

* `docstring : str`
    The action docstring, to be displayed in the status bar when hovering over the action
    item in the menu. By default, the function's docstring.

* `menu : str`
    The name of the menu where the action should be added. It is automatically created
    if it doesn't exist.

* `submenu : str`
    The name of the submenu where the action should be added. It is automatically created
    if it doesn't exist.

* `view : QWidget`
    A view that belongs to the GUI, if the actions are to be added to the view's menu bar.

* `view_submenu : str`
    The name of a submenu in the view menu.

* `checkable : boolean`
    Whether the action is checkable (toggle on/off).

* `checked : boolean`
    Whether the checkable action is initially checked or not.

* `show_shortcut : boolean`
    Whether to show the shortcut in the Help action that displays all GUI shortcuts.

* `icon : str`
    Hexadecimal code of the font-awesome icon.

* `toolbar : boolean`
    Whether to add the action to the toolbar.

---

#### Actions.disable


**`Actions.disable(self, name=None)`**

Disable all actions, or only one if a name is passed.

---

#### Actions.enable


**`Actions.enable(self, name=None)`**

Enable all actions, or only one if a name is passed..

---

#### Actions.get


**`Actions.get(self, name)`**

Get a QAction instance from its name.

---

#### Actions.remove


**`Actions.remove(self, name)`**

Remove an action.

---

#### Actions.remove_all


**`Actions.remove_all(self)`**

Remove all actions.

---

#### Actions.run


**`Actions.run(self, name, *args)`**

Run an action as specified by its name.

---

#### Actions.separator


**`Actions.separator(self, **kwargs)`**

Add a separator.

**Parameters**


* `menu : str`
    The name of the menu where the separator should be added. It is automatically created
    if it doesn't exist.

* `submenu : str`
    The name of the submenu where the separator should be added. It is automatically
    created if it doesn't exist.

* `view : QWidget`
    A view that belongs to the GUI, if the separator is to be added to the view's menu bar.

* `view_submenu : str`
    The name of a submenu in the view menu.

---

#### Actions.show_shortcuts


**`Actions.show_shortcuts(self)`**

Display all shortcuts in the console.

---

#### Actions.shortcuts


**`Actions.shortcuts`**

A dictionary mapping action names to keyboard shortcuts.

---

### phy.gui.Debouncer

Debouncer to work in a Qt application.

Jobs are submitted at given times. They are executed immediately if the
delay since the last submission is greater than some threshold. Otherwise, execution
is delayed until the delay since the last submission is greater than the threshold.
During the waiting time, all submitted jobs erase previous jobs in the queue, so
only the last jobs are taken into account.

This is used when multiple row selections are done in an HTML table, and each row
selection is taking a perceptible time to finish.

**Constructor**


* `delay : int`
    The minimal delay between the execution of two successive actions.

**Example**

```python
d = Debouncer(delay=250)
for i in range(10):
    d.submit(print, "hello world", i)
d.trigger()  # show "hello world 0" and "hello world 9" after a delay

```

---

#### Debouncer.stop_waiting


**`Debouncer.stop_waiting(self, delay=0.1)`**

Stop waiting and force the pending actions to execute (almost) immediately.

---

#### Debouncer.submit


**`Debouncer.submit(self, f, *args, key=None, **kwargs)`**

Submit a function call. Execute immediately if the delay since the last submission
is higher than the threshold, or wait until executing it otherwiser.

---

#### Debouncer.trigger


**`Debouncer.trigger(self)`**

Execute the pending actions.

---

### phy.gui.DockWidget

A dock widget with a custom title bar.

The title bar has a status text at the middle, and a group of buttons on the right.
By default, the buttons on the right are screenshot and close. New buttons can be added
in this group, from right to left.

---

#### DockWidget.add_button


**`DockWidget.add_button(self, callback=None, text=None, icon=None, checkable=False, checked=False, event=None, name=None)`**

Add a button to the dock title bar, to the right.

**Parameters**


* `callback : function`
    Callback function when the button is clicked.

* `text : str`
    Text of the button.

* `icon : str`
    Fontawesome icon of the button specified as a unicode string with 4 hexadecimal
    characters.

* `checkable : boolean`
    Whether the button is checkable.

* `checked : boolean`
    Whether the checkable button is initially checked.

* `event : str`
    Name of the event that is externally raised when the status of the button is changed.
    This is used to synchronize the button's checked status when the value changes
    via another mean than clicking on the button.

* `name : str`
    Name of the button.

---

#### DockWidget.add_checkbox


**`DockWidget.add_checkbox(self, callback=None, text=None, checked=False, name=None)`**

Add a checkbox to the dock title bar, to the right.

**Parameters**


* `callback : function`
    Callback function when the checkbox is clicked.

* `text : str`
    Text of the checkbox.

* `checked : boolean`
    Whether the checkbox is initially checked.

* `name : str`
    Name of the button.

---

#### DockWidget.closeEvent


**`DockWidget.closeEvent(self, e)`**

Qt slot when the window is closed.

---

#### DockWidget.get_widget


**`DockWidget.get_widget(self, name)`**

Get a dock title bar widget by its name.

---

#### DockWidget.set_status


**`DockWidget.set_status(self, text)`**

Set the status text of the widget.

---

#### DockWidget.status


**`DockWidget.status`**

Current status text of the title bar.

---

### phy.gui.GUI

A Qt main window containing docking widgets. This class derives from `QMainWindow`.

**Constructor**


* `position : 2-tuple`
    Coordinates of the GUI window on the screen, in pixels.

* `size : 2-tuple`
    Requested size of the GUI window, in pixels.

* `name : str`
    Name of the GUI window, set in the title bar.

* `subtitle : str`
    Subtitle of the GUI window, set in the title bar after the name.

* `view_creator : dict`
    Map view classnames to functions that take no arguments and return a new view instance
    of that class.

* `view_count : dict`
    Map view classnames to integers specifying the number of views to create for every
    view class.

* `default_views : list-like`
    List of view names to create by default (overriden by `view_count` if not empty).

* `config_dir : str or Path`
    User configuration directory used to load/save the GUI state

* `enable_threading : boolean`
    Whether to enable threading in views or not (used in `ManualClusteringView`).

**Events**

close(gui)
show(gui)
close_view(view, gui)

---

#### GUI.add_view


**`GUI.add_view(self, view, position=None, closable=True, floatable=True, floating=None)`**

Add a dock widget to the main window.

**Parameters**


* `view : View`

* `position : str`
    Relative position where to add the view (left, right, top, bottom).

* `closable : boolean`
    Whether the view can be closed by the user.

* `floatable : boolean`
    Whether the view can be detached from the main GUI.

* `floating : boolean`
    Whether the view should be added in floating mode or not.

---

#### GUI.closeEvent


**`GUI.closeEvent(self, e)`**

Qt slot when the window is closed.

---

#### GUI.create_and_add_view


**`GUI.create_and_add_view(self, view_name)`**

Create a view and add it to the GUI.

---

#### GUI.create_views


**`GUI.create_views(self)`**

Create and add as many views as specified in view_count.

---

#### GUI.dialog


**`GUI.dialog(self, message)`**

Show a message in a dialog box.

---

#### GUI.get_menu


**`GUI.get_menu(self, name, insert_before=None)`**

Get or create a menu.

---

#### GUI.get_submenu


**`GUI.get_submenu(self, menu, name)`**

Get or create a submenu.

---

#### GUI.get_view


**`GUI.get_view(self, cls, index=0)`**

Return a view from a given class. If there are multiple views of the same class,
specify the view index (0 by default).

---

#### GUI.list_views


**`GUI.list_views(self, *classes)`**

Return the list of views which are instances of one or several classes.

---

#### GUI.lock_status


**`GUI.lock_status(self)`**

Lock the status bar.

---

#### GUI.remove_menu


**`GUI.remove_menu(self, name)`**

Remove a menu.

---

#### GUI.restore_geometry_state


**`GUI.restore_geometry_state(self, gs)`**

Restore the position of the main window and the docks.

The GUI widgets need to be recreated first.

This function can be called in `on_show()`.

---

#### GUI.save_geometry_state


**`GUI.save_geometry_state(self)`**

Return picklable geometry and state of the window and docks.

This function can be called in `on_close()`.

---

#### GUI.set_default_actions


**`GUI.set_default_actions(self)`**

Create the default actions (file, views, help...).

---

#### GUI.show


**`GUI.show(self)`**

Show the window.

---

#### GUI.unlock_status


**`GUI.unlock_status(self)`**

Unlock the status bar.

---

#### GUI.status_message


**`GUI.status_message`**

The message in the status bar, can be set by the user.

---

#### GUI.view_count


**`GUI.view_count`**

Return the number of views of every type, as a dictionary mapping view class names
to an integer.

---

#### GUI.views


**`GUI.views`**

Return the list of views in the GUI.

---

### phy.gui.GUIState

Represent the state of the GUI: positions of the views and all parameters associated
to the GUI and views. Derive from `Bunch`, which itself derives from `dict`.

The GUI state is automatically loaded from the user configuration directory.
The default path is `~/.phy/GUIName/state.json`.

The global GUI state is common to all instances of the GUI.
The local GUI state is specific to an instance of the GUI, for example a given dataset.

**Constructor**


* `path : str or Path`
    The path to the JSON file containing the global GUI state.

* `local_path : str or Path`
    The path to the JSON file containing the local GUI state.

* `default_state_path : str or Path`
    The path to the default JSON file provided in the library.

* `local_keys : list`
    A list of strings `key1.key2` of the elements of the GUI state that should only be saved
    in the local state, and not the global state.

---

#### GUIState.add_local_keys


**`GUIState.add_local_keys(self, keys)`**

Add local keys.

---

#### GUIState.copy


**`GUIState.copy(self)`**

Return a new Bunch instance which is a copy of the current Bunch instance.

---

#### GUIState.get_view_state


**`GUIState.get_view_state(self, view)`**

Return the state of a view instance.

---

#### GUIState.load


**`GUIState.load(self)`**

Load the state from the JSON file in the config dir.

---

#### GUIState.save


**`GUIState.save(self)`**

Save the state to the JSON files in the config dir (global) and local dir (if any).

---

#### GUIState.update_view_state


**`GUIState.update_view_state(self, view, state)`**

Update the state of a view instance.

**Parameters**


* `view : View instance`

* `state : Bunch instance`

---

### phy.gui.HTMLBuilder

Build an HTML widget.

---

#### HTMLBuilder.add_header


**`HTMLBuilder.add_header(self, s)`**

Add HTML headers.

---

#### HTMLBuilder.add_script


**`HTMLBuilder.add_script(self, s)`**

Add Javascript code.

---

#### HTMLBuilder.add_script_src


**`HTMLBuilder.add_script_src(self, filename)`**

Add a link to a Javascript file.

---

#### HTMLBuilder.add_style


**`HTMLBuilder.add_style(self, s)`**

Add a CSS style.

---

#### HTMLBuilder.add_style_src


**`HTMLBuilder.add_style_src(self, filename)`**

Add a link to a stylesheet URL.

---

#### HTMLBuilder.set_body


**`HTMLBuilder.set_body(self, body)`**

Set the HTML body of the widget.

---

#### HTMLBuilder.set_body_src


**`HTMLBuilder.set_body_src(self, filename)`**

Set the path to an HTML file containing the body of the widget.

---

#### HTMLBuilder.html


**`HTMLBuilder.html`**

Return the reconstructed HTML code of the widget.

---

### phy.gui.HTMLWidget

An HTML widget that is displayed with Qt, with Javascript support and Python-Javascript
interactions capabilities. These interactions are asynchronous in Qt5, which requires
extensive use of callback functions in Python, as well as synchronization primitives
for unit tests.

**Constructor**


* `parent : Widget`

* `title : window title`

* `debounce_events : list-like`
    The list of event names, raised by the underlying HTML widget, that should be debounced.

---

#### HTMLWidget.build


**`HTMLWidget.build(self, callback=None)`**

Rebuild the HTML code of the widget.

---

#### HTMLWidget.eval_js


**`HTMLWidget.eval_js(self, expr, callback=None)`**

Evaluate a Javascript expression.

**Parameters**


* `expr : str`
    A Javascript expression.

* `callback : function`
    A Python function that is called once the Javascript expression has been
    evaluated. It takes as input the output of the Javascript expression.

---

#### HTMLWidget.set_html


**`HTMLWidget.set_html(self, html, callback=None)`**

Set the HTML code.

---

#### HTMLWidget.view_source


**`HTMLWidget.view_source(self, callback=None)`**

View the HTML source of the widget.

---

#### HTMLWidget.debouncer


**`HTMLWidget.debouncer`**

Widget debouncer.

---

### phy.gui.IPythonView

A view with an IPython console living in the same Python process as the GUI.

---

#### IPythonView.attach


**`IPythonView.attach(self, gui, **kwargs)`**

Add the view to the GUI, start the kernel, and inject the specified variables.

---

#### IPythonView.inject


**`IPythonView.inject(self, **kwargs)`**

Inject variables into the IPython namespace.

---

#### IPythonView.start_kernel


**`IPythonView.start_kernel(self)`**

Start the IPython kernel.

---

#### IPythonView.stop


**`IPythonView.stop(self)`**

Stop the kernel.

---

### phy.gui.KeyValueWidget

A Qt widget that displays a simple form where each field has a name, a type, and accept
user input.

---

#### KeyValueWidget.add_pair


**`KeyValueWidget.add_pair(self, name, default=None, vtype=None)`**

Add a key-value pair.

**Parameters**


* `name : str`

* `default : object`

* `vtype : str`
    Can be 'str' (text box), 'int' (spin box), 'float' (spin box), 'bool' (checkbox),
    'mutiline' (text edit for multiline str), or 'list' (several widgets).

---

#### KeyValueWidget.attach


**`KeyValueWidget.attach(self, gui)`**

Add the view to a GUI.

---

#### KeyValueWidget.get_value


**`KeyValueWidget.get_value(self, name)`**

Get the default or user-entered value of a field.

---

#### KeyValueWidget.get_widget


**`KeyValueWidget.get_widget(self, name)`**

Get the widget of a field.

---

#### KeyValueWidget.to_dict


**`KeyValueWidget.to_dict(self)`**

Return the key-value mapping dictionary as specified by the user inputs and defaults.

---

#### KeyValueWidget.names


**`KeyValueWidget.names`**

List of field names.

---

### phy.gui.Snippets

Provide keyboard snippets to quickly execute actions from a GUI.

This class attaches to a GUI and an `Actions` instance. To every command
is associated a snippet with the same name, or with an alias as indicated
in the action. The arguments of the action's callback functions can be
provided in the snippet's command with a simple syntax. For example, the
following command:

```
:my_action string 3-6
```

corresponds to:

```python
my_action('string', (3, 4, 5, 6))
```

The snippet mode is activated with the `:` keyboard shortcut. A snippet
command is activated with `Enter`, and one can leave the snippet mode
with `Escape`.

When the snippet mode is enabled (with `:`), this object adds a hidden Qt action
for every keystroke. These actions are removed when the snippet mode is disabled.

**Constructor**


* `gui : GUI instance`

---

#### Snippets.is_mode_on


**`Snippets.is_mode_on(self)`**

Whether the snippet mode is enabled.

---

#### Snippets.mode_off


**`Snippets.mode_off(self)`**

Disable the snippet mode.

---

#### Snippets.mode_on


**`Snippets.mode_on(self)`**

Enable the snippet mode.

---

#### Snippets.run


**`Snippets.run(self, snippet)`**

Execute a snippet command.

May be overridden.

---

#### Snippets.command


**`Snippets.command`**

This is used to write a snippet message in the status bar. A cursor is appended at
the end.

---

### phy.gui.Table

A sortable table with support for selection. Derives from HTMLWidget.

This table uses the following Javascript implementation: https://github.com/kwikteam/tablejs
This Javascript class builds upon ListJS: https://listjs.com/

---

#### Table.add


**`Table.add(self, objects)`**

Add objects object to the table.

---

#### Table.build


**`Table.build(self, callback=None)`**

Rebuild the HTML code of the widget.

---

#### Table.change


**`Table.change(self, objects)`**

Change some objects.

---

#### Table.eval_js


**`Table.eval_js(self, expr, callback=None)`**

Evaluate a Javascript expression.

The `table` Javascript variable can be used to interact with the underlying Javascript
table.

The table has sortable columns, a filter text box, support for single and multi selection
of rows. Rows can be skippable (used for ignored clusters in phy).

The table can raise Javascript events that are relayed to Python. Objects are
transparently serialized and deserialized in JSON. Basic types (numbers, strings, lists)
are transparently converted between Python and Javascript.

**Parameters**


* `expr : str`
    A Javascript expression.

* `callback : function`
    A Python function that is called once the Javascript expression has been
    evaluated. It takes as input the output of the Javascript expression.

---

#### Table.filter


**`Table.filter(self, text='')`**

Filter the view with a Javascript expression.

---

#### Table.first


**`Table.first(self, callback=None)`**

Select the first item.

---

#### Table.get


**`Table.get(self, id, callback=None)`**

Get the object given its id.

---

#### Table.get_current_sort


**`Table.get_current_sort(self, callback=None)`**

Get the current sort as a tuple `(name, dir)`.

---

#### Table.get_ids


**`Table.get_ids(self, callback=None)`**

Get the list of ids.

---

#### Table.get_next_id


**`Table.get_next_id(self, callback=None)`**

Get the next non-skipped row id.

---

#### Table.get_previous_id


**`Table.get_previous_id(self, callback=None)`**

Get the previous non-skipped row id.

---

#### Table.get_selected


**`Table.get_selected(self, callback=None)`**

Get the currently selected rows.

---

#### Table.is_ready


**`Table.is_ready(self)`**

Whether the widget has been fully loaded.

---

#### Table.last


**`Table.last(self, callback=None)`**

Select the last item.

---

#### Table.next


**`Table.next(self, callback=None)`**

Select the next non-skipped row.

---

#### Table.previous


**`Table.previous(self, callback=None)`**

Select the previous non-skipped row.

---

#### Table.remove


**`Table.remove(self, ids)`**

Remove some objects from their ids.

---

#### Table.remove_all


**`Table.remove_all(self)`**

Remove all rows in the table.

---

#### Table.remove_all_and_add


**`Table.remove_all_and_add(self, objects)`**

Remove all rows in the table and add new objects.

---

#### Table.scroll_to


**`Table.scroll_to(self, id)`**

Scroll until a given row is visible.

---

#### Table.select


**`Table.select(self, ids, callback=None, **kwargs)`**

Select some rows in the table from Python.

This function calls `table.select()` in Javascript, which raises a Javascript event
relayed to Python. This sequence of actions is the same when the user selects
rows directly in the HTML view.

---

#### Table.set_busy


**`Table.set_busy(self, busy)`**

Set the busy state of the GUI.

---

#### Table.set_html


**`Table.set_html(self, html, callback=None)`**

Set the HTML code.

---

#### Table.sort_by


**`Table.sort_by(self, name, sort_dir='asc')`**

Sort by a given variable.

---

#### Table.view_source


**`Table.view_source(self, callback=None)`**

View the HTML source of the widget.

---

#### Table.debouncer


**`Table.debouncer`**

Widget debouncer.

---

### phy.gui.Worker

A task (just a Python function) running in the thread pool.

**Constructor**


* `fn : function`

* `*args : function positional arguments`

* `**kwargs : function keyword arguments`

---

#### Worker.run


**`Worker.run(self)`**

Run the task in a background thread. Should not be called directly unless you want
to bypass the thread pool.

---

## phy.plot

Plotting module based on OpenGL.

For advanced users!

---

#### phy.plot.Subplot


**`phy.plot.Subplot(shape=None, index=None, shape_gpu_var=None, index_gpu_var=None)`**

Return a particular Range transform that transforms from NDC to a subplot at a particular
location, in a grid layout.

**Parameters**

* `shape : 2-tuple`
    Number of rows and columns in the grid layout.

* `index : 2-tuple`
    Index o the row and column of the subplot.

* `shape_gpu_var : str`
    Name of the GPU variable with the grid's shape.

* `index_gpu_var : str`
    Name of the GPU variable with the grid's subplot index.

---

#### phy.plot.extend_bounds


**`phy.plot.extend_bounds(bounds_list)`**

Return a single data bounds 4-tuple from a list of data bounds.

---

#### phy.plot.get_linear_x


**`phy.plot.get_linear_x(n_signals, n_samples)`**

Get a vertical stack of arrays ranging from -1 to 1.

Return a `(n_signals, n_samples)` array.

---

### phy.plot.Axes

Dynamic axes that move along the camera when panning and zooming.

**Constructor**


* `data_bounds : 4-tuple`
    The data coordinates of the initial viewport (when there is no panning and zooming).

* `color : 4-tuple`
    Color of the grid.

* `show_x : boolean`
    Whether to show the vertical grid lines.

* `show_y : boolean`
    Whether to show the horizontal grid lines.

---

#### Axes.attach


**`Axes.attach(self, canvas)`**

Add the axes to a canvas.

Add the grid and text visuals to the canvas, and attach to the pan and zoom events
raised by the canvas.

---

#### Axes.reset_data_bounds


**`Axes.reset_data_bounds(self, data_bounds, do_update=True)`**

Reset the bounds of the view in data coordinates.

Used when the view is recreated from scratch.

---

#### Axes.update_visuals


**`Axes.update_visuals(self)`**

Update the grid and text visuals after updating the axis locator.

---

### phy.plot.AxisLocator

Determine the location of ticks in a view.

**Constructor**


* `nbinsx : int`
    Number of ticks on the x axis.

* `nbinsy : int`
    Number of ticks on the y axis.

* `data_bounds : 4-tuple`
    Initial coordinates of the viewport, as (xmin, ymin, xmax, ymax), in data coordinates.
    These are the data coordinates of the lower left and upper right points of the window.

---

#### AxisLocator.set_nbins


**`AxisLocator.set_nbins(self, nbinsx=None, nbinsy=None)`**

Change the number of bins on the x and y axes.

---

#### AxisLocator.set_view_bounds


**`AxisLocator.set_view_bounds(self, view_bounds=None)`**

Set the view bounds in normalized device coordinates. Used when panning and zooming.

This method updates the following attributes:

* xticks : the position of the ticks on the x axis
* yticks : the position of the ticks on the y axis
* xtext : the text of the ticks on the x axis
* ytext : the text of the ticks on the y axis

---

### phy.plot.BaseCanvas

Base canvas class. Derive from QOpenGLWindow.

The canvas represents an OpenGL-powered rectangular black window where one can add visuals
and attach interaction (pan/zoom, lasso) and layout (subplot) compaion objects.

---

#### BaseCanvas.add_visual


**`BaseCanvas.add_visual(self, visual, **kwargs)`**

Add a visual to the canvas and build its OpenGL program using the attached interacts.

We can't build the visual's program before, because we need the canvas' transforms first.

**Parameters**


* `visual : Visual`

* `clearable : True`
    Whether the visual should be deleted when calling `canvas.clear()`.

* `exclude_origins : list-like`
    List of interact instances that should not apply to that visual. For example, use to
    add a visual outside of the subplots, or with no support for pan and zoom.

* `key : str`
    An optional key to identify a visual

---

#### BaseCanvas.attach_events


**`BaseCanvas.attach_events(self, obj)`**

Attach an object that has `on_xxx()` methods. These methods are called when internal
events are raised by the canvas. This is used for mouse and key interactions.

---

#### BaseCanvas.clear


**`BaseCanvas.clear(self)`**

Remove all visuals except those marked `clearable=False`.

---

#### BaseCanvas.emit


**`BaseCanvas.emit(self, name, **kwargs)`**

Raise an internal event and call `on_xxx()` on attached objects.

---

#### BaseCanvas.event


**`BaseCanvas.event(self, e)`**

Touch event.

---

#### BaseCanvas.get_size


**`BaseCanvas.get_size(self)`**

Return the window size in pixels.

---

#### BaseCanvas.get_visual


**`BaseCanvas.get_visual(self, key)`**

Get a visual from its key.

---

#### BaseCanvas.has_visual


**`BaseCanvas.has_visual(self, visual)`**

Return whether a visual belongs to the canvas.

---

#### BaseCanvas.initializeGL


**`BaseCanvas.initializeGL(self)`**

Create the scene.

---

#### BaseCanvas.iter_update_queue


**`BaseCanvas.iter_update_queue(self)`**

Iterate through all OpenGL program updates called in lazy mode.

---

#### BaseCanvas.keyPressEvent


**`BaseCanvas.keyPressEvent(self, e)`**

Emit an internal `key_press` event.

---

#### BaseCanvas.keyReleaseEvent


**`BaseCanvas.keyReleaseEvent(self, e)`**

Emit an internal `key_release` event.

---

#### BaseCanvas.mouseDoubleClickEvent


**`BaseCanvas.mouseDoubleClickEvent(self, e)`**

Emit an internal `mouse_double_click` event.

---

#### BaseCanvas.mouseMoveEvent


**`BaseCanvas.mouseMoveEvent(self, e)`**

Emit an internal `mouse_move` event.

---

#### BaseCanvas.mousePressEvent


**`BaseCanvas.mousePressEvent(self, e)`**

Emit an internal `mouse_press` event.

---

#### BaseCanvas.mouseReleaseEvent


**`BaseCanvas.mouseReleaseEvent(self, e)`**

Emit an internal `mouse_release` or `mouse_click` event.

---

#### BaseCanvas.on_next_paint


**`BaseCanvas.on_next_paint(self, f)`**

Register a function to be called at the next frame refresh (in paintGL()).

---

#### BaseCanvas.paintGL


**`BaseCanvas.paintGL(self)`**

Draw all visuals.

---

#### BaseCanvas.remove


**`BaseCanvas.remove(self, *visuals)`**

Remove some visuals objects from the canvas.

---

#### BaseCanvas.resizeEvent


**`BaseCanvas.resizeEvent(self, e)`**

Emit a `resize(width, height)` event when resizing the window.

---

#### BaseCanvas.set_lazy


**`BaseCanvas.set_lazy(self, lazy)`**

When the lazy mode is enabled, all OpenGL calls are deferred. Use with
multithreading.

Must be called *after* the visuals have been added, but *before* set_data().

---

#### BaseCanvas.update


**`BaseCanvas.update(self)`**

Update the OpenGL canvas.

---

#### BaseCanvas.wheelEvent


**`BaseCanvas.wheelEvent(self, e)`**

Emit an internal `mouse_wheel` event.

---

#### BaseCanvas.window_to_ndc


**`BaseCanvas.window_to_ndc(self, mouse_pos)`**

Convert a mouse position in pixels into normalized device coordinates, taking into
account pan and zoom.

---

### phy.plot.BaseLayout

Implement global transforms on a canvas, like subplots.

---

#### BaseLayout.attach


**`BaseLayout.attach(self, canvas)`**

Attach this layout to a canvas.

---

#### BaseLayout.box_map


**`BaseLayout.box_map(self, mouse_pos)`**

Get the box and local NDC coordinates from mouse position.

---

#### BaseLayout.get_closest_box


**`BaseLayout.get_closest_box(self, ndc)`**

Override to return the box closest to a given position in NDC.

---

#### BaseLayout.imap


**`BaseLayout.imap(self, arr, box=None)`**

Apply the layout inverse transformation to a position array.

---

#### BaseLayout.map


**`BaseLayout.map(self, arr, box=None, inverse=None)`**

Apply the layout transformation to a position array.

---

#### BaseLayout.swap_active_box


**`BaseLayout.swap_active_box(self, box)`**

Context manager to temporary change the active box.

---

#### BaseLayout.update


**`BaseLayout.update(self)`**

Update all visuals in the attached canvas.

---

#### BaseLayout.update_visual


**`BaseLayout.update_visual(self, visual)`**

Called whenever visual.set_data() is called. Set a_box_index in here.

---

### phy.plot.BaseVisual

A Visual represents one object (or homogeneous set of objects).

It is rendered with a single pass of a single gloo program with a single type of GL primitive.

**Main abstract methods**

validate
    takes as input the visual's parameters, set the default values, and validates all
    values
vertex_count
    takes as input the visual's parameters, and return the total number of vertices
set_data
    takes as input the visual's parameters, and ends with update calls to the underlying
    OpenGL program: `self.program[name] = data`

**Notes**

* set_data MUST set self.n_vertices (necessary for a_box_index in layouts)
* set_data MUST call `self.emit_visual_set_data()` at the end, and return the data

---

#### BaseVisual.add_batch_data


**`BaseVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### BaseVisual.close


**`BaseVisual.close(self)`**

Close the visual.

---

#### BaseVisual.emit_visual_set_data


**`BaseVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### BaseVisual.hide


**`BaseVisual.hide(self)`**

Hide the visual.

---

#### BaseVisual.on_draw


**`BaseVisual.on_draw(self)`**

Draw the visual.

---

#### BaseVisual.on_resize


**`BaseVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### BaseVisual.reset_batch


**`BaseVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### BaseVisual.set_box_index


**`BaseVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### BaseVisual.set_data


**`BaseVisual.set_data(self)`**

Set data to the program.

Must be called *after* attach(canvas), because the program is built
when the visual is attached to the canvas.

---

#### BaseVisual.set_data_range


**`BaseVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### BaseVisual.set_primitive_type


**`BaseVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### BaseVisual.set_shader


**`BaseVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### BaseVisual.show


**`BaseVisual.show(self)`**

Show the visual.

---

#### BaseVisual.toggle


**`BaseVisual.toggle(self)`**

Toggle the visual visibility.

---

#### BaseVisual.validate


**`BaseVisual.validate(**kwargs)`**

Make consistent the input data for the visual.

---

#### BaseVisual.vertex_count


**`BaseVisual.vertex_count(**kwargs)`**

Return the number of vertices as a function of the input data.

---

### phy.plot.BatchAccumulator

Accumulate data arrays for batch visuals.

This class is used to simplify the creation of batch visuals, where different visual elements
of the same type are concatenated into a singual Visual instance, which significantly
improves the performance of OpenGL.

---

#### BatchAccumulator.add


**`BatchAccumulator.add(self, b, noconcat=(), n_items=None, n_vertices=None, **kwargs)`**

Add data for a given batch iteration.

**Parameters**


* `b : Bunch`
    Data to add to the current batch iteration.

* `noconcat : tuple`
    List of keys that should not be concatenated.

* `n_items : int`
    Number of visual items to add in this batch iteration.

* `n_vertices : int`
    Number of vertices added in this batch iteration.

**Note**

`n_items` and `n_vertices` differ for special visuals, like `TextVisual` where each
item is a string, but is represented in OpenGL as a number of vertices (six times the
number of characters, as each character requires two triangles).

---

#### BatchAccumulator.reset


**`BatchAccumulator.reset(self)`**

Reset the accumulator.

---

#### BatchAccumulator.data


**`BatchAccumulator.data`**

Return the concatenated data as a dictionary.

---

### phy.plot.Boxed

Layout showing plots in rectangles at arbitrary positions. Used by the waveform view.

The boxes are specified via their center positions and optional sizes, in which case
an iterative algorithm is used to find the largest box size that will not make them overlap.

**Constructor**


* `box_pos : array-like (2D, shape[1] == 2)`
    Position of the centers of the boxes.

* `box_var : str`
    Name of the GLSL variable with the box index.

* `keep_aspect_ratio : boolean`
    Whether to keep the aspect ratio of the bounds.

**Note**

To be used in a boxed layout, a visual must define `a_box_index` (by default) or another GLSL
variable specified in `box_var`.

---

#### Boxed.add_boxes


**`Boxed.add_boxes(self, canvas)`**

Show the boxes borders.

---

#### Boxed.attach


**`Boxed.attach(self, canvas)`**

Attach the boxed interact to a canvas.

---

#### Boxed.box_map


**`Boxed.box_map(self, mouse_pos)`**

Get the box and local NDC coordinates from mouse position.

---

#### Boxed.expand_box_height


**`Boxed.expand_box_height(self)`**



---

#### Boxed.expand_box_width


**`Boxed.expand_box_width(self)`**



---

#### Boxed.expand_layout_height


**`Boxed.expand_layout_height(self)`**



---

#### Boxed.expand_layout_width


**`Boxed.expand_layout_width(self)`**



---

#### Boxed.get_closest_box


**`Boxed.get_closest_box(self, pos)`**

Get the box closest to some position.

---

#### Boxed.imap


**`Boxed.imap(self, arr, box=None)`**

Apply the layout inverse transformation to a position array.

---

#### Boxed.map


**`Boxed.map(self, arr, box=None, inverse=None)`**

Apply the layout transformation to a position array.

---

#### Boxed.shrink_box_height


**`Boxed.shrink_box_height(self)`**



---

#### Boxed.shrink_box_width


**`Boxed.shrink_box_width(self)`**



---

#### Boxed.shrink_layout_height


**`Boxed.shrink_layout_height(self)`**



---

#### Boxed.shrink_layout_width


**`Boxed.shrink_layout_width(self)`**



---

#### Boxed.swap_active_box


**`Boxed.swap_active_box(self, box)`**

Context manager to temporary change the active box.

---

#### Boxed.update


**`Boxed.update(self)`**

Update all visuals in the attached canvas.

---

#### Boxed.update_boxes


**`Boxed.update_boxes(self, box_pos)`**

Update the box positions and automatically-computed size.

---

#### Boxed.update_visual


**`Boxed.update_visual(self, visual)`**

Update a visual.

---

#### Boxed.box_bounds


**`Boxed.box_bounds`**

Bounds of the boxes.

---

#### Boxed.box_scaling


**`Boxed.box_scaling`**



---

#### Boxed.layout_scaling


**`Boxed.layout_scaling`**



---

#### Boxed.n_boxes


**`Boxed.n_boxes`**

Total number of boxes.

---

### phy.plot.GLSLInserter

Object used to insert GLSL snippets into shader code.

This class provides methods to specify the snippets to insert, and the
`insert_into_shaders()` method inserts them into a vertex and fragment shader.

---

#### GLSLInserter.add_gpu_transforms


**`GLSLInserter.add_gpu_transforms(self, tc)`**

Insert all GLSL snippets from a transform chain.

---

#### GLSLInserter.add_varying


**`GLSLInserter.add_varying(self, vtype, name, value)`**

Add a varying variable.

---

#### GLSLInserter.insert_frag


**`GLSLInserter.insert_frag(self, glsl, location=None, origin=None, index=None)`**

Insert a GLSL snippet into the fragment shader. See `insert_vert()`.

---

#### GLSLInserter.insert_into_shaders


**`GLSLInserter.insert_into_shaders(self, vertex, fragment, exclude_origins=())`**

Insert all GLSL snippets in a vertex and fragment shaders.

**Parameters**


* `vertex : str`
    GLSL code of the vertex shader

* `fragment : str`
    GLSL code of the fragment shader

* `exclude_origins : list-like`
    List of interact instances to exclude when inserting the shaders.

**Notes**

The vertex shader typicall contains `gl_Position = transform(data_var_name);`
which is automatically detected, and the GLSL transformations are inserted there.

Snippets can contain `{{var}}` placeholders for the transformed variable name.

---

#### GLSLInserter.insert_vert


**`GLSLInserter.insert_vert(self, glsl, location='transforms', origin=None, index=None)`**

Insert a GLSL snippet into the vertex shader.

**Parameters**


* `glsl : str`
    The GLSL code to insert.

* `location : str`
    Where to insert the GLSL code. Can be:

    * `header`: declaration of GLSL variables
    * `start`: start of the function
    * `before_transforms`: just before the transforms in the vertex shader
    * `transforms`: where the GPU transforms are applied in the vertex shader
    * `after_transforms`: just after the GPU transforms
    * `end`: end of the function


* `origin : Interact`
    The interact object that adds this GLSL snippet. Should be discared by
    visuals that are added with that interact object in `exclude_origins`.

* `index : int`
    Index of the snippets list to insert the snippet.

---

### phy.plot.Grid

Layout showing subplots arranged in a 2D grid.

**Constructor**


* `shape : tuple or str`
    Number of rows, cols in the grid.

* `shape_var : str`
    Name of the GLSL uniform variable that holds the shape, when it is variable.

* `box_var : str`
    Name of the GLSL variable with the box index.

* `has_clip : boolean`
    Whether subplots should be clipped.

**Note**

To be used in a grid, a visual must define `a_box_index` (by default) or another GLSL
variable specified in `box_var`.

---

#### Grid.add_boxes


**`Grid.add_boxes(self, canvas, shape=None)`**

Show subplot boxes.

---

#### Grid.attach


**`Grid.attach(self, canvas)`**

Attach the grid to a canvas.

---

#### Grid.box_map


**`Grid.box_map(self, mouse_pos)`**

Get the box and local NDC coordinates from mouse position.

---

#### Grid.get_closest_box


**`Grid.get_closest_box(self, pos)`**

Get the box index (i, j) closest to a given position in NDC coordinates.

---

#### Grid.imap


**`Grid.imap(self, arr, box=None)`**

Apply the layout inverse transformation to a position array.

---

#### Grid.map


**`Grid.map(self, arr, box=None, inverse=None)`**

Apply the layout transformation to a position array.

---

#### Grid.swap_active_box


**`Grid.swap_active_box(self, box)`**

Context manager to temporary change the active box.

---

#### Grid.update


**`Grid.update(self)`**

Update all visuals in the attached canvas.

---

#### Grid.update_visual


**`Grid.update_visual(self, visual)`**

Update a visual.

---

#### Grid.scaling


**`Grid.scaling`**

Return the grid scaling.

---

#### Grid.shape


**`Grid.shape`**

Return the grid shape.

---

### phy.plot.HistogramVisual

A histogram visual.

**Parameters**


* `hist : array-like (1D), or list of 1D arrays, or 2D array`

* `color : array-like (2D, shape[1] == 4)`

* `ylim : array-like (1D)`
    The maximum hist value in the viewport.

---

#### HistogramVisual.add_batch_data


**`HistogramVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### HistogramVisual.close


**`HistogramVisual.close(self)`**

Close the visual.

---

#### HistogramVisual.emit_visual_set_data


**`HistogramVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### HistogramVisual.hide


**`HistogramVisual.hide(self)`**

Hide the visual.

---

#### HistogramVisual.on_draw


**`HistogramVisual.on_draw(self)`**

Draw the visual.

---

#### HistogramVisual.on_resize


**`HistogramVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### HistogramVisual.reset_batch


**`HistogramVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### HistogramVisual.set_box_index


**`HistogramVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### HistogramVisual.set_data


**`HistogramVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---

#### HistogramVisual.set_data_range


**`HistogramVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### HistogramVisual.set_primitive_type


**`HistogramVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### HistogramVisual.set_shader


**`HistogramVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### HistogramVisual.show


**`HistogramVisual.show(self)`**

Show the visual.

---

#### HistogramVisual.toggle


**`HistogramVisual.toggle(self)`**

Toggle the visual visibility.

---

#### HistogramVisual.validate


**`HistogramVisual.validate(self, hist=None, color=None, ylim=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---

#### HistogramVisual.vertex_count


**`HistogramVisual.vertex_count(self, hist, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.ImageVisual

Display a 2D image.

**Parameters**

* `image : array-like (3D)`

---

#### ImageVisual.add_batch_data


**`ImageVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### ImageVisual.close


**`ImageVisual.close(self)`**

Close the visual.

---

#### ImageVisual.emit_visual_set_data


**`ImageVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### ImageVisual.hide


**`ImageVisual.hide(self)`**

Hide the visual.

---

#### ImageVisual.on_draw


**`ImageVisual.on_draw(self)`**

Draw the visual.

---

#### ImageVisual.on_resize


**`ImageVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### ImageVisual.reset_batch


**`ImageVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### ImageVisual.set_box_index


**`ImageVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### ImageVisual.set_data


**`ImageVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---

#### ImageVisual.set_data_range


**`ImageVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### ImageVisual.set_primitive_type


**`ImageVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### ImageVisual.set_shader


**`ImageVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### ImageVisual.show


**`ImageVisual.show(self)`**

Show the visual.

---

#### ImageVisual.toggle


**`ImageVisual.toggle(self)`**

Toggle the visual visibility.

---

#### ImageVisual.validate


**`ImageVisual.validate(self, image=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---

#### ImageVisual.vertex_count


**`ImageVisual.vertex_count(self, image=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.Lasso

Draw a polygon with the mouse and find the points that belong to the inside of the
polygon.

---

#### Lasso.add


**`Lasso.add(self, pos)`**

Add a point to the polygon.

---

#### Lasso.attach


**`Lasso.attach(self, canvas)`**

Attach the lasso to a canvas.

---

#### Lasso.clear


**`Lasso.clear(self)`**

Reset the lasso.

---

#### Lasso.create_lasso_visual


**`Lasso.create_lasso_visual(self)`**

Create the lasso visual.

---

#### Lasso.in_polygon


**`Lasso.in_polygon(self, pos)`**

Return which points belong to the polygon.

---

#### Lasso.on_mouse_click


**`Lasso.on_mouse_click(self, e)`**

Add a polygon point with ctrl+click.

---

#### Lasso.update_lasso_visual


**`Lasso.update_lasso_visual(self)`**

Update the lasso visual with the current polygon.

---

#### Lasso.count


**`Lasso.count`**

Number of vertices in the polygon.

---

#### Lasso.polygon


**`Lasso.polygon`**

Coordinates of the polygon vertices.

---

### phy.plot.LineVisual

Line segments.

**Parameters**

* `pos : array-like (2D)`

* `color : array-like (2D, shape[1] == 4)`

* `data_bounds : array-like (2D, shape[1] == 4)`

---

#### LineVisual.add_batch_data


**`LineVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### LineVisual.close


**`LineVisual.close(self)`**

Close the visual.

---

#### LineVisual.emit_visual_set_data


**`LineVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### LineVisual.hide


**`LineVisual.hide(self)`**

Hide the visual.

---

#### LineVisual.on_draw


**`LineVisual.on_draw(self)`**

Draw the visual.

---

#### LineVisual.on_resize


**`LineVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### LineVisual.reset_batch


**`LineVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### LineVisual.set_box_index


**`LineVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### LineVisual.set_data


**`LineVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---

#### LineVisual.set_data_range


**`LineVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### LineVisual.set_primitive_type


**`LineVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### LineVisual.set_shader


**`LineVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### LineVisual.show


**`LineVisual.show(self)`**

Show the visual.

---

#### LineVisual.toggle


**`LineVisual.toggle(self)`**

Toggle the visual visibility.

---

#### LineVisual.validate


**`LineVisual.validate(self, pos=None, color=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---

#### LineVisual.vertex_count


**`LineVisual.vertex_count(self, pos=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.PanZoom

Pan and zoom interact. Support mouse and keyboard interactivity.

**Constructor**


* `aspect : float`
    Aspect ratio to keep while panning and zooming.

* `pan : 2-tuple`
    Initial pan.

* `zoom : 2-tuple`
    Initial zoom.

* `zmin : float`
    Minimum zoom allowed.

* `zmax : float`
    Maximum zoom allowed.

* `xmin : float`
    Minimum x allowed.

* `xmax : float`
    Maximum x allowed.

* `ymin : float`
    Minimum y allowed.

* `ymax : float`
    Maximum y allowed.

* `constrain_bounds : 4-tuple`
    Equivalent to (xmin, ymin, xmax, ymax).

* `pan_var_name : str`
    Name of the pan GLSL variable name

* `zoom_var_name : str`
    Name of the zoom GLSL variable name

* `enable_mouse_wheel : boolean`
    Whether to enable the mouse wheel for zooming.

**Interactivity**

* Keyboard arrows for panning
* Keyboard + and - for zooming
* Mouse left button + drag for panning
* Mouse right button + drag for zooming
* Mouse wheel for zooming
* R and double-click for reset

**Example**

```python

# Create and attach the PanZoom interact.
pz = PanZoom()
pz.attach(canvas)

# Create a visual.
visual = MyVisual(...)
visual.set_data(...)

# Attach the visual to the canvas.
canvas = BaseCanvas()
visual.attach(canvas, 'PanZoom')

canvas.show()
```

---

#### PanZoom.attach


**`PanZoom.attach(self, canvas)`**

Attach this interact to a canvas.

---

#### PanZoom.emit_update_events


**`PanZoom.emit_update_events(self)`**

Emit the pan and zoom events to update views after a pan zoom manual update.

---

#### PanZoom.get_range


**`PanZoom.get_range(self)`**

Return the bounds currently visible.

---

#### PanZoom.imap


**`PanZoom.imap(self, arr)`**

Apply the current panzoom inverse transformation to a position array.

---

#### PanZoom.map


**`PanZoom.map(self, arr)`**

Apply the current panzoom transformation to a position array.

---

#### PanZoom.on_key_press


**`PanZoom.on_key_press(self, e)`**

Pan and zoom with the keyboard.

---

#### PanZoom.on_mouse_double_click


**`PanZoom.on_mouse_double_click(self, e)`**

Reset the view by double clicking anywhere in the canvas.

---

#### PanZoom.on_mouse_move


**`PanZoom.on_mouse_move(self, e)`**

Pan and zoom with the mouse.

---

#### PanZoom.on_mouse_wheel


**`PanZoom.on_mouse_wheel(self, e)`**

Zoom with the mouse wheel.

---

#### PanZoom.on_resize


**`PanZoom.on_resize(self, e)`**

Resize event.

---

#### PanZoom.pan_delta


**`PanZoom.pan_delta(self, d)`**

Pan the view by a given amount.

---

#### PanZoom.reset


**`PanZoom.reset(self)`**

Reset the view.

---

#### PanZoom.set_constrain_bounds


**`PanZoom.set_constrain_bounds(self, bounds)`**



---

#### PanZoom.set_pan_zoom


**`PanZoom.set_pan_zoom(self, pan=None, zoom=None)`**

Set at once the pan and zoom.

---

#### PanZoom.set_range


**`PanZoom.set_range(self, bounds, keep_aspect=False)`**

Zoom to fit a box.

---

#### PanZoom.update


**`PanZoom.update(self)`**

Update all visuals in the attached canvas.

---

#### PanZoom.update_visual


**`PanZoom.update_visual(self, visual)`**

Update a visual with the current pan and zoom values.

---

#### PanZoom.window_to_ndc


**`PanZoom.window_to_ndc(self, pos)`**

Return the mouse coordinates in NDC, taking panzoom into account.

---

#### PanZoom.zoom_delta


**`PanZoom.zoom_delta(self, d, p=(0.0, 0.0), c=1.0)`**

Zoom the view by a given amount.

---

#### PanZoom.aspect


**`PanZoom.aspect`**

Aspect (width/height).

---

#### PanZoom.pan


**`PanZoom.pan`**

Pan translation.

---

#### PanZoom.size


**`PanZoom.size`**

Window size of the canvas.

---

#### PanZoom.xmax


**`PanZoom.xmax`**

Maximum x allowed for pan.

---

#### PanZoom.xmin


**`PanZoom.xmin`**

Minimum x allowed for pan.

---

#### PanZoom.ymax


**`PanZoom.ymax`**

Maximum y allowed for pan.

---

#### PanZoom.ymin


**`PanZoom.ymin`**

Minimum y allowed for pan.

---

#### PanZoom.zmax


**`PanZoom.zmax`**

Maximal zoom level.

---

#### PanZoom.zmin


**`PanZoom.zmin`**

Minimum zoom level.

---

#### PanZoom.zoom


**`PanZoom.zoom`**

Zoom level.

---

### phy.plot.PlotCanvas

Plotting canvas that supports different layouts, subplots, lasso, axes, panzoom.

---

#### PlotCanvas.add_visual


**`PlotCanvas.add_visual(self, visual, *args, **kwargs)`**

Add a visual and possibly set some data directly.

**Parameters**


* `visual : Visual`

* `clearable : True`
    Whether the visual should be deleted when calling `canvas.clear()`.

* `exclude_origins : list-like`
    List of interact instances that should not apply to that visual. For example, use to
    add a visual outside of the subplots, or with no support for pan and zoom.

* `key : str`
    An optional key to identify a visual

---

#### PlotCanvas.attach_events


**`PlotCanvas.attach_events(self, obj)`**

Attach an object that has `on_xxx()` methods. These methods are called when internal
events are raised by the canvas. This is used for mouse and key interactions.

---

#### PlotCanvas.clear


**`PlotCanvas.clear(self)`**

Remove all visuals except those marked `clearable=False`.

---

#### PlotCanvas.emit


**`PlotCanvas.emit(self, name, **kwargs)`**

Raise an internal event and call `on_xxx()` on attached objects.

---

#### PlotCanvas.enable_axes


**`PlotCanvas.enable_axes(self, data_bounds=None, show_x=True, show_y=True)`**

Show axes in the canvas.

---

#### PlotCanvas.enable_lasso


**`PlotCanvas.enable_lasso(self)`**

Enable lasso in the canvas.

---

#### PlotCanvas.enable_panzoom


**`PlotCanvas.enable_panzoom(self)`**

Enable pan zoom in the canvas.

---

#### PlotCanvas.event


**`PlotCanvas.event(self, e)`**

Touch event.

---

#### PlotCanvas.get_size


**`PlotCanvas.get_size(self)`**

Return the window size in pixels.

---

#### PlotCanvas.get_visual


**`PlotCanvas.get_visual(self, key)`**

Get a visual from its key.

---

#### PlotCanvas.has_visual


**`PlotCanvas.has_visual(self, visual)`**

Return whether a visual belongs to the canvas.

---

#### PlotCanvas.hist


**`PlotCanvas.hist(self, *args, **kwargs)`**

Add a standalone (no batch) histogram plot.

---

#### PlotCanvas.initializeGL


**`PlotCanvas.initializeGL(self)`**

Create the scene.

---

#### PlotCanvas.iter_update_queue


**`PlotCanvas.iter_update_queue(self)`**

Iterate through all OpenGL program updates called in lazy mode.

---

#### PlotCanvas.keyPressEvent


**`PlotCanvas.keyPressEvent(self, e)`**

Emit an internal `key_press` event.

---

#### PlotCanvas.keyReleaseEvent


**`PlotCanvas.keyReleaseEvent(self, e)`**

Emit an internal `key_release` event.

---

#### PlotCanvas.lines


**`PlotCanvas.lines(self, *args, **kwargs)`**

Add a standalone (no batch) line plot.

---

#### PlotCanvas.mouseDoubleClickEvent


**`PlotCanvas.mouseDoubleClickEvent(self, e)`**

Emit an internal `mouse_double_click` event.

---

#### PlotCanvas.mouseMoveEvent


**`PlotCanvas.mouseMoveEvent(self, e)`**

Emit an internal `mouse_move` event.

---

#### PlotCanvas.mousePressEvent


**`PlotCanvas.mousePressEvent(self, e)`**

Emit an internal `mouse_press` event.

---

#### PlotCanvas.mouseReleaseEvent


**`PlotCanvas.mouseReleaseEvent(self, e)`**

Emit an internal `mouse_release` or `mouse_click` event.

---

#### PlotCanvas.on_next_paint


**`PlotCanvas.on_next_paint(self, f)`**

Register a function to be called at the next frame refresh (in paintGL()).

---

#### PlotCanvas.paintGL


**`PlotCanvas.paintGL(self)`**

Draw all visuals.

---

#### PlotCanvas.plot


**`PlotCanvas.plot(self, *args, **kwargs)`**

Add a standalone (no batch) plot.

---

#### PlotCanvas.polygon


**`PlotCanvas.polygon(self, *args, **kwargs)`**

Add a standalone (no batch) polygon plot.

---

#### PlotCanvas.remove


**`PlotCanvas.remove(self, *visuals)`**

Remove some visuals objects from the canvas.

---

#### PlotCanvas.resizeEvent


**`PlotCanvas.resizeEvent(self, e)`**

Emit a `resize(width, height)` event when resizing the window.

---

#### PlotCanvas.scatter


**`PlotCanvas.scatter(self, *args, **kwargs)`**

Add a standalone (no batch) scatter plot.

---

#### PlotCanvas.set_layout


**`PlotCanvas.set_layout(self, layout=None, shape=None, n_plots=None, origin=None, box_pos=None, has_clip=True)`**

Set the plot layout: grid, boxed, stacked, or None.

---

#### PlotCanvas.set_lazy


**`PlotCanvas.set_lazy(self, lazy)`**

When the lazy mode is enabled, all OpenGL calls are deferred. Use with
multithreading.

Must be called *after* the visuals have been added, but *before* set_data().

---

#### PlotCanvas.text


**`PlotCanvas.text(self, *args, **kwargs)`**

Add a standalone (no batch) text plot.

---

#### PlotCanvas.update


**`PlotCanvas.update(self)`**

Update the OpenGL canvas.

---

#### PlotCanvas.update_visual


**`PlotCanvas.update_visual(self, visual, *args, **kwargs)`**

Set the data of a visual, standalone or at the end of a batch.

---

#### PlotCanvas.uplot


**`PlotCanvas.uplot(self, *args, **kwargs)`**

Add a standalone (no batch) uniform plot.

---

#### PlotCanvas.uscatter


**`PlotCanvas.uscatter(self, *args, **kwargs)`**

Add a standalone (no batch) uniform scatter plot.

---

#### PlotCanvas.wheelEvent


**`PlotCanvas.wheelEvent(self, e)`**

Emit an internal `mouse_wheel` event.

---

#### PlotCanvas.window_to_ndc


**`PlotCanvas.window_to_ndc(self, mouse_pos)`**

Convert a mouse position in pixels into normalized device coordinates, taking into
account pan and zoom.

---

#### PlotCanvas.canvas


**`PlotCanvas.canvas`**



---

### phy.plot.PlotVisual

Plot visual, with multiple line plots of various sizes and colors.

**Parameters**


* `x : array-like (1D), or list of 1D arrays for different plots`

* `y : array-like (1D), or list of 1D arrays, for different plots`

* `color : array-like (2D, shape[-1] == 4)`

* `depth : array-like (1D)`

* `masks : array-like (1D)`
    Similar to an alpha channel, but for color saturation instead of transparency.

* `data_bounds : array-like (2D, shape[1] == 4)`

---

#### PlotVisual.add_batch_data


**`PlotVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### PlotVisual.close


**`PlotVisual.close(self)`**

Close the visual.

---

#### PlotVisual.emit_visual_set_data


**`PlotVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### PlotVisual.hide


**`PlotVisual.hide(self)`**

Hide the visual.

---

#### PlotVisual.on_draw


**`PlotVisual.on_draw(self)`**

Draw the visual.

---

#### PlotVisual.on_resize


**`PlotVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### PlotVisual.reset_batch


**`PlotVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### PlotVisual.set_box_index


**`PlotVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### PlotVisual.set_color


**`PlotVisual.set_color(self, color)`**

Update the visual's color.

---

#### PlotVisual.set_data


**`PlotVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---

#### PlotVisual.set_data_range


**`PlotVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### PlotVisual.set_primitive_type


**`PlotVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### PlotVisual.set_shader


**`PlotVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### PlotVisual.show


**`PlotVisual.show(self)`**

Show the visual.

---

#### PlotVisual.toggle


**`PlotVisual.toggle(self)`**

Toggle the visual visibility.

---

#### PlotVisual.validate


**`PlotVisual.validate(self, x=None, y=None, color=None, depth=None, masks=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---

#### PlotVisual.vertex_count


**`PlotVisual.vertex_count(self, y=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.PolygonVisual

Polygon.

**Parameters**

* `pos : array-like (2D)`

* `data_bounds : array-like (2D, shape[1] == 4)`

---

#### PolygonVisual.add_batch_data


**`PolygonVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### PolygonVisual.close


**`PolygonVisual.close(self)`**

Close the visual.

---

#### PolygonVisual.emit_visual_set_data


**`PolygonVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### PolygonVisual.hide


**`PolygonVisual.hide(self)`**

Hide the visual.

---

#### PolygonVisual.on_draw


**`PolygonVisual.on_draw(self)`**

Draw the visual.

---

#### PolygonVisual.on_resize


**`PolygonVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### PolygonVisual.reset_batch


**`PolygonVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### PolygonVisual.set_box_index


**`PolygonVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### PolygonVisual.set_data


**`PolygonVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---

#### PolygonVisual.set_data_range


**`PolygonVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### PolygonVisual.set_primitive_type


**`PolygonVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### PolygonVisual.set_shader


**`PolygonVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### PolygonVisual.show


**`PolygonVisual.show(self)`**

Show the visual.

---

#### PolygonVisual.toggle


**`PolygonVisual.toggle(self)`**

Toggle the visual visibility.

---

#### PolygonVisual.validate


**`PolygonVisual.validate(self, pos=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---

#### PolygonVisual.vertex_count


**`PolygonVisual.vertex_count(self, pos=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.Range

Linear transform from a source rectangle to a target rectangle.

**Constructor**


* `from_bounds : 4-tuple`
    Bounds of the source rectangle.

* `to_bounds : 4-tuple`
    Bounds of the target rectangle.

* `from_gpu_var : str`
    Name of the GPU variable with the from bounds.

* `to_gpu_var : str`
    Name of the GPU variable with the to bounds.

---

#### Range.apply


**`Range.apply(self, arr, from_bounds=None, to_bounds=None)`**

Apply the transform to a NumPy array.

---

#### Range.glsl


**`Range.glsl(self, var)`**

Return a GLSL snippet that applies the transform to a given GLSL variable name.

---

#### Range.inverse


**`Range.inverse(self)`**

Return the inverse Range instance.

---

### phy.plot.Scale

Scale transform.

**Constructor**

* `amount : 2-tuple`
    Coordinates of the scaling.

* `gpu_var : str`
    The name of the GPU variable with the scaling vector.

---

#### Scale.apply


**`Scale.apply(self, arr, param=None)`**

Apply a scaling to a NumPy array.

---

#### Scale.glsl


**`Scale.glsl(self, var)`**

Return a GLSL snippet that applies the scaling to a given GLSL variable name.

---

#### Scale.inverse


**`Scale.inverse(self)`**

Return the inverse Scale instance.

---

### phy.plot.ScatterVisual

Scatter visual, displaying a fixed marker at various positions, colors, and marker sizes.

**Constructor**


* `marker : string (used for all points in the scatter visual)`
    Default: disc. Can be one of: arrow, asterisk, chevron, clover, club, cross, diamond,
    disc, ellipse, hbar, heart, infinity, pin, ring, spade, square, tag, triangle, vbar

**Parameters**


* `x : array-like (1D)`

* `y : array-like (1D)`

* `pos : array-like (2D)`

* `color : array-like (2D, shape[1] == 4)`

* `size : array-like (1D)`
    Marker sizes, in pixels

* `depth : array-like (1D)`

* `data_bounds : array-like (2D, shape[1] == 4)`

---

#### ScatterVisual.add_batch_data


**`ScatterVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### ScatterVisual.close


**`ScatterVisual.close(self)`**

Close the visual.

---

#### ScatterVisual.emit_visual_set_data


**`ScatterVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### ScatterVisual.hide


**`ScatterVisual.hide(self)`**

Hide the visual.

---

#### ScatterVisual.on_draw


**`ScatterVisual.on_draw(self)`**

Draw the visual.

---

#### ScatterVisual.on_resize


**`ScatterVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### ScatterVisual.reset_batch


**`ScatterVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### ScatterVisual.set_box_index


**`ScatterVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### ScatterVisual.set_color


**`ScatterVisual.set_color(self, color)`**

Change the color of the markers.

---

#### ScatterVisual.set_data


**`ScatterVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---

#### ScatterVisual.set_data_range


**`ScatterVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### ScatterVisual.set_marker_size


**`ScatterVisual.set_marker_size(self, marker_size)`**

Change the size of the markers.

---

#### ScatterVisual.set_primitive_type


**`ScatterVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### ScatterVisual.set_shader


**`ScatterVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### ScatterVisual.show


**`ScatterVisual.show(self)`**

Show the visual.

---

#### ScatterVisual.toggle


**`ScatterVisual.toggle(self)`**

Toggle the visual visibility.

---

#### ScatterVisual.validate


**`ScatterVisual.validate(self, x=None, y=None, pos=None, color=None, size=None, depth=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---

#### ScatterVisual.vertex_count


**`ScatterVisual.vertex_count(self, x=None, y=None, pos=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.TextVisual

Display strings at multiple locations.

**Constructor**


* `color : 4-tuple`

* `font_size : float`
    The font size, in points (8 by default).

**Parameters**


* `pos : array-like (2D)`
    Position of each string (of variable length).

* `text : list of strings (variable lengths)`

* `anchor : array-like (2D)`
    For each string, specifies the anchor of the string with respect to the string's position.

    Examples:

    * (0, 0): text centered at the position
    * (1, 1): the position is at the lower left of the string
    * (1, -1): the position is at the upper left of the string
    * (-1, 1): the position is at the lower right of the string
    * (-1, -1): the position is at the upper right of the string

    Values higher than 1 or lower than -1 can be used as margins, knowing that the unit
    of the anchor is (string width, string height).


* `data_bounds : array-like (2D, shape[1] == 4)`

---

#### TextVisual.add_batch_data


**`TextVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### TextVisual.close


**`TextVisual.close(self)`**

Close the visual.

---

#### TextVisual.emit_visual_set_data


**`TextVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### TextVisual.hide


**`TextVisual.hide(self)`**

Hide the visual.

---

#### TextVisual.on_draw


**`TextVisual.on_draw(self)`**

Draw the visual.

---

#### TextVisual.on_resize


**`TextVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### TextVisual.reset_batch


**`TextVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### TextVisual.set_box_index


**`TextVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### TextVisual.set_data


**`TextVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---

#### TextVisual.set_data_range


**`TextVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### TextVisual.set_primitive_type


**`TextVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### TextVisual.set_shader


**`TextVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### TextVisual.show


**`TextVisual.show(self)`**

Show the visual.

---

#### TextVisual.toggle


**`TextVisual.toggle(self)`**

Toggle the visual visibility.

---

#### TextVisual.validate


**`TextVisual.validate(self, pos=None, text=None, color=None, anchor=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---

#### TextVisual.vertex_count


**`TextVisual.vertex_count(self, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.TransformChain

A linear sequence of transforms.

---

#### TransformChain.add


**`TransformChain.add(self, transforms, origin=None)`**

Add some transforms.

---

#### TransformChain.apply


**`TransformChain.apply(self, arr)`**

Apply all transforms on an array.

---

#### TransformChain.get


**`TransformChain.get(self, class_name)`**

Get a transform in the chain from its name.

---

#### TransformChain.inverse


**`TransformChain.inverse(self)`**

Return the inverse chain of transforms.

---

#### TransformChain.transforms


**`TransformChain.transforms`**

List of transforms.

---

### phy.plot.Translate

Translation transform.

**Constructor**

* `amount : 2-tuple`
    Coordinates of the translation.

* `gpu_var : str`
    The name of the GPU variable with the translate vector.

---

#### Translate.apply


**`Translate.apply(self, arr, param=None)`**

Apply a translation to a NumPy array.

---

#### Translate.glsl


**`Translate.glsl(self, var)`**

Return a GLSL snippet that applies the translation to a given GLSL variable name.

---

#### Translate.inverse


**`Translate.inverse(self)`**

Return the inverse Translate instance.

---

### phy.plot.UniformPlotVisual

A plot visual with a uniform color.

**Constructor**


* `color : 4-tuple`

* `depth : scalar`

**Parameters**


* `x : array-like (1D), or list of 1D arrays for different plots`

* `y : array-like (1D), or list of 1D arrays, for different plots`

* `masks : array-like (1D)`
    Similar to an alpha channel, but for color saturation instead of transparency.

* `data_bounds : array-like (2D, shape[1] == 4)`

---

#### UniformPlotVisual.add_batch_data


**`UniformPlotVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### UniformPlotVisual.close


**`UniformPlotVisual.close(self)`**

Close the visual.

---

#### UniformPlotVisual.emit_visual_set_data


**`UniformPlotVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### UniformPlotVisual.hide


**`UniformPlotVisual.hide(self)`**

Hide the visual.

---

#### UniformPlotVisual.on_draw


**`UniformPlotVisual.on_draw(self)`**

Draw the visual.

---

#### UniformPlotVisual.on_resize


**`UniformPlotVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### UniformPlotVisual.reset_batch


**`UniformPlotVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### UniformPlotVisual.set_box_index


**`UniformPlotVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### UniformPlotVisual.set_data


**`UniformPlotVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---

#### UniformPlotVisual.set_data_range


**`UniformPlotVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### UniformPlotVisual.set_primitive_type


**`UniformPlotVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### UniformPlotVisual.set_shader


**`UniformPlotVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### UniformPlotVisual.show


**`UniformPlotVisual.show(self)`**

Show the visual.

---

#### UniformPlotVisual.toggle


**`UniformPlotVisual.toggle(self)`**

Toggle the visual visibility.

---

#### UniformPlotVisual.validate


**`UniformPlotVisual.validate(self, x=None, y=None, masks=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---

#### UniformPlotVisual.vertex_count


**`UniformPlotVisual.vertex_count(self, y=None, **kwargs)`**

Number of vertices for the requested data.

---

### phy.plot.UniformScatterVisual

Scatter visual with a fixed marker color and size.

**Constructor**


* `marker : str`

* `color : 4-tuple`

* `size : scalar`

**Parameters**


* `x : array-like (1D)`

* `y : array-like (1D)`

* `pos : array-like (2D)`

* `masks : array-like (1D)`
    Similar to an alpha channel, but for color saturation instead of transparency.

* `data_bounds : array-like (2D, shape[1] == 4)`

---

#### UniformScatterVisual.add_batch_data


**`UniformScatterVisual.add_batch_data(self, **kwargs)`**

Prepare data to be added later with `PlotCanvas.add_visual()`.

---

#### UniformScatterVisual.close


**`UniformScatterVisual.close(self)`**

Close the visual.

---

#### UniformScatterVisual.emit_visual_set_data


**`UniformScatterVisual.emit_visual_set_data(self)`**

Emit canvas.visual_set_data event after data has been set in the visual.

---

#### UniformScatterVisual.hide


**`UniformScatterVisual.hide(self)`**

Hide the visual.

---

#### UniformScatterVisual.on_draw


**`UniformScatterVisual.on_draw(self)`**

Draw the visual.

---

#### UniformScatterVisual.on_resize


**`UniformScatterVisual.on_resize(self, width, height)`**

Update the window size in the OpenGL program.

---

#### UniformScatterVisual.reset_batch


**`UniformScatterVisual.reset_batch(self)`**

Reinitialize the batch.

---

#### UniformScatterVisual.set_box_index


**`UniformScatterVisual.set_box_index(self, box_index, data=None)`**

Set the visual's box index. This is used by layouts (e.g. subplot indices).

---

#### UniformScatterVisual.set_data


**`UniformScatterVisual.set_data(self, *args, **kwargs)`**

Update the visual data.

---

#### UniformScatterVisual.set_data_range


**`UniformScatterVisual.set_data_range(self, data_range)`**

Add a CPU Range transform for data normalization.

---

#### UniformScatterVisual.set_primitive_type


**`UniformScatterVisual.set_primitive_type(self, primitive_type)`**

Set the primitive type (points, lines, line_strip, line_fan, triangles).

---

#### UniformScatterVisual.set_shader


**`UniformScatterVisual.set_shader(self, name)`**

Set the built-in vertex and fragment shader.

---

#### UniformScatterVisual.show


**`UniformScatterVisual.show(self)`**

Show the visual.

---

#### UniformScatterVisual.toggle


**`UniformScatterVisual.toggle(self)`**

Toggle the visual visibility.

---

#### UniformScatterVisual.validate


**`UniformScatterVisual.validate(self, x=None, y=None, pos=None, masks=None, data_bounds=None, **kwargs)`**

Validate the requested data before passing it to set_data().

---

#### UniformScatterVisual.vertex_count


**`UniformScatterVisual.vertex_count(self, x=None, y=None, pos=None, **kwargs)`**

Number of vertices for the requested data.

---

## phy.cluster

Manual clustering facilities.

---

#### phy.cluster.select_traces


**`phy.cluster.select_traces(traces, interval, sample_rate=None)`**

Load traces in an interval (in seconds).

---

### phy.cluster.AmplitudeView

This view displays an amplitude plot for all selected clusters.

**Constructor**


* `amplitudes : dict`
    Dictionary `{amplitudes_type: function}`, for different types of amplitudes.

    Each function maps `cluster_ids` to a list
    `[Bunch(amplitudes, spike_ids, spike_times), ...]` for each cluster.
    Use `cluster_id=None` for background amplitudes.

---

#### AmplitudeView.attach


**`AmplitudeView.attach(self, gui)`**

Attach the view to the GUI.

---

#### AmplitudeView.close


**`AmplitudeView.close(self)`**

Close the view.

---

#### AmplitudeView.decrease_marker_size


**`AmplitudeView.decrease_marker_size(self)`**

Decrease the scaling parameter.

---

#### AmplitudeView.get_clusters_data


**`AmplitudeView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

---

#### AmplitudeView.increase_marker_size


**`AmplitudeView.increase_marker_size(self)`**

Increase the scaling parameter.

---

#### AmplitudeView.next_amplitudes_type


**`AmplitudeView.next_amplitudes_type(self)`**

Switch to the next amplitudes type.

---

#### AmplitudeView.on_cluster


**`AmplitudeView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### AmplitudeView.on_mouse_click


**`AmplitudeView.on_mouse_click(self, e)`**

Select a time from the amplitude view to display in the trace view.

---

#### AmplitudeView.on_mouse_wheel


**`AmplitudeView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### AmplitudeView.on_request_split


**`AmplitudeView.on_request_split(self, sender=None)`**

Return the spikes enclosed by the lasso.

---

#### AmplitudeView.on_select


**`AmplitudeView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### AmplitudeView.on_select_threaded


**`AmplitudeView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### AmplitudeView.plot


**`AmplitudeView.plot(self, **kwargs)`**

Update the view with the current cluster selection.

---

#### AmplitudeView.previous_amplitudes_type


**`AmplitudeView.previous_amplitudes_type(self)`**

Switch to the previous amplitudes type.

---

#### AmplitudeView.reset_marker_size


**`AmplitudeView.reset_marker_size(self)`**

Reset the scaling to the default value.

---

#### AmplitudeView.screenshot


**`AmplitudeView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### AmplitudeView.set_state


**`AmplitudeView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### AmplitudeView.show


**`AmplitudeView.show(self)`**

Show the underlying canvas.

---

#### AmplitudeView.show_time_range


**`AmplitudeView.show_time_range(self, interval=(0, 0))`**



---

#### AmplitudeView.toggle_auto_update


**`AmplitudeView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### AmplitudeView.update_status


**`AmplitudeView.update_status(self)`**



---

#### AmplitudeView.amplitudes_type


**`AmplitudeView.amplitudes_type`**



---

#### AmplitudeView.marker_size


**`AmplitudeView.marker_size`**

Size of the spike markers, in pixels.

---

#### AmplitudeView.state


**`AmplitudeView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### AmplitudeView.status


**`AmplitudeView.status`**



---

### phy.cluster.ClusterMeta

Handle cluster metadata changes.

---

#### ClusterMeta.add_field


**`ClusterMeta.add_field(self, name, default_value=None)`**

Add a field with an optional default value.

---

#### ClusterMeta.from_dict


**`ClusterMeta.from_dict(self, dic)`**

Import data from a `{cluster_id: {field: value}}` dictionary.

---

#### ClusterMeta.get


**`ClusterMeta.get(self, field, cluster)`**

Retrieve the value of one cluster for a given field.

---

#### ClusterMeta.redo


**`ClusterMeta.redo(self)`**

Redo the next metadata change.

**Returns**


* `up : UpdateInfo instance`

---

#### ClusterMeta.set


**`ClusterMeta.set(self, field, clusters, value, add_to_stack=True)`**

Set the value of one of several clusters.

**Parameters**


* `field : str`
    The field to set.

* `clusters : list`
    The list of cluster ids to change.

* `value : str`
    The new metadata value for the given clusters.

* `add_to_stack : boolean`
    Whether this metadata change should be recorded in the undo stack.

**Returns**


* `up : UpdateInfo instance`

---

#### ClusterMeta.set_from_descendants


**`ClusterMeta.set_from_descendants(self, descendants, largest_old_cluster=None)`**

Update metadata of some clusters given the metadata of their ascendants.

**Parameters**


* `descendants : list`
    List of pairs (old_cluster_id, new_cluster_id)

* `largest_old_cluster : int`
    If available, the cluster id of the largest old cluster, used as a reference.

---

#### ClusterMeta.to_dict


**`ClusterMeta.to_dict(self, field)`**

Export data to a `{cluster_id: value}` dictionary, for a particular field.

---

#### ClusterMeta.undo


**`ClusterMeta.undo(self)`**

Undo the last metadata change.

**Returns**


* `up : UpdateInfo instance`

---

#### ClusterMeta.fields


**`ClusterMeta.fields`**

List of fields.

---

### phy.cluster.ClusterScatterView

This view shows all clusters in a customizable scatter plot.

**Constructor**


* `cluster_ids : array-like`
cluster_info: function
    Maps cluster_id => Bunch() with attributes.
bindings: dict
    Maps plot dimension to cluster attributes.

---

#### ClusterScatterView.add_color_scheme


**`ClusterScatterView.add_color_scheme(self, fun=None, name=None, cluster_ids=None, colormap=None, categorical=None, logarithmic=None)`**

Add a color scheme to the view. Can be used as follows:

```python
@connect
def on_view_attached(gui, view):
    view.add_color_scheme(c.get_depth, name='depth', colormap='linear')
```

---

#### ClusterScatterView.attach


**`ClusterScatterView.attach(self, gui)`**

Attach the GUI.

---

#### ClusterScatterView.change_bindings


**`ClusterScatterView.change_bindings(self, **kwargs)`**

Change the bindings.

---

#### ClusterScatterView.close


**`ClusterScatterView.close(self)`**

Close the view.

---

#### ClusterScatterView.decrease_marker_size


**`ClusterScatterView.decrease_marker_size(self)`**

Decrease the scaling parameter.

---

#### ClusterScatterView.get_cluster_colors


**`ClusterScatterView.get_cluster_colors(self, cluster_ids, alpha=1.0)`**

Return the cluster colors depending on the currently-selected color scheme.

---

#### ClusterScatterView.get_cluster_data


**`ClusterScatterView.get_cluster_data(self, cluster_id)`**

Return the data of one cluster.

---

#### ClusterScatterView.get_clusters_data


**`ClusterScatterView.get_clusters_data(self, cluster_ids)`**

Return the data of a set of clusters, as a dictionary {cluster_id: Bunch}.

---

#### ClusterScatterView.increase_marker_size


**`ClusterScatterView.increase_marker_size(self)`**

Increase the scaling parameter.

---

#### ClusterScatterView.next_color_scheme


**`ClusterScatterView.next_color_scheme(self)`**

Switch to the next color scheme.

---

#### ClusterScatterView.on_cluster


**`ClusterScatterView.on_cluster(self, sender, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### ClusterScatterView.on_mouse_click


**`ClusterScatterView.on_mouse_click(self, e)`**

Select a cluster by clicking on its template waveform.

---

#### ClusterScatterView.on_mouse_wheel


**`ClusterScatterView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### ClusterScatterView.on_select


**`ClusterScatterView.on_select(self, *args, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### ClusterScatterView.on_select_threaded


**`ClusterScatterView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### ClusterScatterView.plot


**`ClusterScatterView.plot(self, **kwargs)`**

Make the scatter plot.

---

#### ClusterScatterView.prepare_color


**`ClusterScatterView.prepare_color(self)`**

Compute the marker colors.

---

#### ClusterScatterView.prepare_data


**`ClusterScatterView.prepare_data(self)`**

Prepare the marker position, size, and color from the cluster information.

---

#### ClusterScatterView.prepare_position


**`ClusterScatterView.prepare_position(self)`**

Compute the marker positions.

---

#### ClusterScatterView.prepare_size


**`ClusterScatterView.prepare_size(self)`**

Compute the marker sizes.

---

#### ClusterScatterView.previous_color_scheme


**`ClusterScatterView.previous_color_scheme(self)`**

Switch to the previous color scheme.

---

#### ClusterScatterView.reset_marker_size


**`ClusterScatterView.reset_marker_size(self)`**

Reset the scaling to the default value.

---

#### ClusterScatterView.screenshot


**`ClusterScatterView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### ClusterScatterView.set_cluster_ids


**`ClusterScatterView.set_cluster_ids(self, all_cluster_ids)`**

Update the cluster data by specifying the list of all cluster ids.

---

#### ClusterScatterView.set_fields


**`ClusterScatterView.set_fields(self)`**



---

#### ClusterScatterView.set_size


**`ClusterScatterView.set_size(self, field)`**

Set the dimension for the marker size.

---

#### ClusterScatterView.set_spike_clusters


**`ClusterScatterView.set_spike_clusters(self, spike_clusters)`**



---

#### ClusterScatterView.set_state


**`ClusterScatterView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### ClusterScatterView.set_x_axis


**`ClusterScatterView.set_x_axis(self, field)`**

Set the dimension for the x axis.

---

#### ClusterScatterView.set_y_axis


**`ClusterScatterView.set_y_axis(self, field)`**

Set the dimension for the y axis.

---

#### ClusterScatterView.show


**`ClusterScatterView.show(self)`**

Show the underlying canvas.

---

#### ClusterScatterView.toggle_auto_update


**`ClusterScatterView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### ClusterScatterView.toggle_log_scale


**`ClusterScatterView.toggle_log_scale(self, dim, checked)`**

Toggle logarithmic scaling for one of the dimensions.

---

#### ClusterScatterView.update_cluster_sort


**`ClusterScatterView.update_cluster_sort(self, cluster_ids)`**



---

#### ClusterScatterView.update_color


**`ClusterScatterView.update_color(self)`**

Update the cluster colors depending on the current color scheme.

---

#### ClusterScatterView.update_select_color


**`ClusterScatterView.update_select_color(self)`**

Update the cluster colors after the cluster selection changes.

---

#### ClusterScatterView.update_status


**`ClusterScatterView.update_status(self)`**



---

#### ClusterScatterView.bindings


**`ClusterScatterView.bindings`**



---

#### ClusterScatterView.color_scheme


**`ClusterScatterView.color_scheme`**

Current color scheme.

---

#### ClusterScatterView.marker_size


**`ClusterScatterView.marker_size`**

Size of the spike markers, in pixels.

---

#### ClusterScatterView.state


**`ClusterScatterView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### ClusterScatterView.status


**`ClusterScatterView.status`**



---

### phy.cluster.ClusterView

Display a table of all clusters with metrics and labels as columns. Derive from Table.

**Constructor**


* `parent : Qt widget`

* `data : list`
    List of dictionaries mapping fields to values.

* `columns : list`
    List of columns in the table.

* `sort : 2-tuple`
    Initial sort of the table as a pair (column_name, order), where order is
    either `asc` or `desc`.

---

#### ClusterView.add


**`ClusterView.add(self, objects)`**

Add objects object to the table.

---

#### ClusterView.build


**`ClusterView.build(self, callback=None)`**

Rebuild the HTML code of the widget.

---

#### ClusterView.change


**`ClusterView.change(self, objects)`**

Change some objects.

---

#### ClusterView.eval_js


**`ClusterView.eval_js(self, expr, callback=None)`**

Evaluate a Javascript expression.

The `table` Javascript variable can be used to interact with the underlying Javascript
table.

The table has sortable columns, a filter text box, support for single and multi selection
of rows. Rows can be skippable (used for ignored clusters in phy).

The table can raise Javascript events that are relayed to Python. Objects are
transparently serialized and deserialized in JSON. Basic types (numbers, strings, lists)
are transparently converted between Python and Javascript.

**Parameters**


* `expr : str`
    A Javascript expression.

* `callback : function`
    A Python function that is called once the Javascript expression has been
    evaluated. It takes as input the output of the Javascript expression.

---

#### ClusterView.filter


**`ClusterView.filter(self, text='')`**

Filter the view with a Javascript expression.

---

#### ClusterView.first


**`ClusterView.first(self, callback=None)`**

Select the first item.

---

#### ClusterView.get


**`ClusterView.get(self, id, callback=None)`**

Get the object given its id.

---

#### ClusterView.get_current_sort


**`ClusterView.get_current_sort(self, callback=None)`**

Get the current sort as a tuple `(name, dir)`.

---

#### ClusterView.get_ids


**`ClusterView.get_ids(self, callback=None)`**

Get the list of ids.

---

#### ClusterView.get_next_id


**`ClusterView.get_next_id(self, callback=None)`**

Get the next non-skipped row id.

---

#### ClusterView.get_previous_id


**`ClusterView.get_previous_id(self, callback=None)`**

Get the previous non-skipped row id.

---

#### ClusterView.get_selected


**`ClusterView.get_selected(self, callback=None)`**

Get the currently selected rows.

---

#### ClusterView.is_ready


**`ClusterView.is_ready(self)`**

Whether the widget has been fully loaded.

---

#### ClusterView.last


**`ClusterView.last(self, callback=None)`**

Select the last item.

---

#### ClusterView.next


**`ClusterView.next(self, callback=None)`**

Select the next non-skipped row.

---

#### ClusterView.previous


**`ClusterView.previous(self, callback=None)`**

Select the previous non-skipped row.

---

#### ClusterView.remove


**`ClusterView.remove(self, ids)`**

Remove some objects from their ids.

---

#### ClusterView.remove_all


**`ClusterView.remove_all(self)`**

Remove all rows in the table.

---

#### ClusterView.remove_all_and_add


**`ClusterView.remove_all_and_add(self, objects)`**

Remove all rows in the table and add new objects.

---

#### ClusterView.scroll_to


**`ClusterView.scroll_to(self, id)`**

Scroll until a given row is visible.

---

#### ClusterView.select


**`ClusterView.select(self, ids, callback=None, **kwargs)`**

Select some rows in the table from Python.

This function calls `table.select()` in Javascript, which raises a Javascript event
relayed to Python. This sequence of actions is the same when the user selects
rows directly in the HTML view.

---

#### ClusterView.set_busy


**`ClusterView.set_busy(self, busy)`**

Set the busy state of the GUI.

---

#### ClusterView.set_html


**`ClusterView.set_html(self, html, callback=None)`**

Set the HTML code.

---

#### ClusterView.set_state


**`ClusterView.set_state(self, state)`**

Set the cluster view state, with a specified sort.

---

#### ClusterView.sort_by


**`ClusterView.sort_by(self, name, sort_dir='asc')`**

Sort by a given variable.

---

#### ClusterView.view_source


**`ClusterView.view_source(self, callback=None)`**

View the HTML source of the widget.

---

#### ClusterView.debouncer


**`ClusterView.debouncer`**

Widget debouncer.

---

#### ClusterView.state


**`ClusterView.state`**

Return the cluster view state, with the current sort and selection.

---

### phy.cluster.Clustering

Handle cluster changes in a set of spikes.

**Constructor**


* `spike_clusters : array-like`
    Spike-cluster assignments, giving the cluster id of every spike.

* `new_cluster_id : int`
    Cluster id that is not used yet (and not used in the cache if there is one). We need to
    ensure that cluster ids are unique and not reused in a given session.

* `spikes_per_cluster : dict`
    Dictionary mapping each cluster id to the spike ids belonging to it. This is recomputed
    if not given. This object may take a while to compute, so it may be cached and passed
    to the constructor.

**Features**

* List of clusters appearing in a `spike_clusters` array
* Dictionary of spikes per cluster
* Merge
* Split and assign
* Undo/redo stack

**Notes**

The undo stack works by keeping the list of all spike cluster changes
made successively. Undoing consists of reapplying all changes from the
original `spike_clusters` array, except the last one.

**UpdateInfo**

Most methods of this class return an `UpdateInfo` instance. This object
contains information about the clustering changes done by the operation.
This object is used throughout the `phy.cluster.manual` package to let
different classes know about clustering changes.

`UpdateInfo` is a dictionary that also supports dot access (`Bunch` class).

---

#### Clustering.assign


**`Clustering.assign(self, spike_ids, spike_clusters_rel=0)`**

Make new spike cluster assignments.

**Parameters**


* `spike_ids : array-like`
    List of spike ids.

* `spike_clusters_rel : array-like`
    Relative cluster ids of the spikes in `spike_ids`. This
    must have the same size as `spike_ids`.

**Returns**


* `up : UpdateInfo instance`

**Note**

`spike_clusters_rel` contain *relative* cluster indices. Their values
don't matter: what matters is whether two give spikes
should end up in the same cluster or not. Adding a constant number
to all elements in `spike_clusters_rel` results in exactly the same
operation.

The final cluster ids are automatically generated by the `Clustering`
class. This is because we must ensure that all modified clusters
get brand new ids. The whole library is based on the assumption that
cluster ids are unique and "disposable". Changing a cluster always
results in a new cluster id being assigned.

If a spike is assigned to a new cluster, then all other spikes
belonging to the same cluster are assigned to a brand new cluster,
even if they were not changed explicitely by the `assign()` method.

In other words, the list of spikes affected by an `assign()` is almost
always a strict superset of the `spike_ids` parameter. The only case
where this is not true is when whole clusters change: this is called
a merge. It is implemented in a separate `merge()` method because it
is logically much simpler, and faster to execute.

---

#### Clustering.merge


**`Clustering.merge(self, cluster_ids, to=None)`**

Merge several clusters to a new cluster.

**Parameters**


* `cluster_ids : array-like`
    List of clusters to merge.

* `to : integer`
    The id of the new cluster. By default, this is `new_cluster_id()`.

**Returns**


* `up : UpdateInfo instance`

---

#### Clustering.new_cluster_id


**`Clustering.new_cluster_id(self)`**

Generate a brand new cluster id.

**Note**

This new id strictly increases after an undo + new action,
meaning that old cluster ids are *not* reused. This ensures that
any cluster_id-based cache will always be valid even after undo
operations (i.e. no need for explicit cache invalidation in this case).

---

#### Clustering.redo


**`Clustering.redo(self)`**

Redo the last cluster assignment operation.

**Returns**


* `up : UpdateInfo instance of the changes done by this operation.`

---

#### Clustering.reset


**`Clustering.reset(self)`**

Reset the clustering to the original clustering.

All changes are lost.

---

#### Clustering.spikes_in_clusters


**`Clustering.spikes_in_clusters(self, clusters)`**

Return the array of spike ids belonging to a list of clusters.

---

#### Clustering.split


**`Clustering.split(self, spike_ids, spike_clusters_rel=0)`**

Split a number of spikes into a new cluster.

This is equivalent to an `assign()` to a single new cluster.

**Parameters**


* `spike_ids : array-like`
    Array of spike ids to split.

* `spike_clusters_rel : array-like (or None)`
    Array of relative spike clusters.

**Returns**


* `up : UpdateInfo instance`

**Note**

The note in the `assign()` method applies here as well. The list
of spikes affected by the split is almost always a strict superset
of the spike_ids parameter.

---

#### Clustering.undo


**`Clustering.undo(self)`**

Undo the last cluster assignment operation.

**Returns**


* `up : UpdateInfo instance of the changes done by this operation.`

---

#### Clustering.cluster_ids


**`Clustering.cluster_ids`**

Ordered list of ids of all non-empty clusters.

---

#### Clustering.n_clusters


**`Clustering.n_clusters`**

Total number of clusters.

---

#### Clustering.n_spikes


**`Clustering.n_spikes`**

Number of spikes.

---

#### Clustering.spike_clusters


**`Clustering.spike_clusters`**

A n_spikes-long vector containing the cluster ids of all spikes.

---

#### Clustering.spike_ids


**`Clustering.spike_ids`**

Array of all spike ids.

---

#### Clustering.spikes_per_cluster


**`Clustering.spikes_per_cluster`**

A dictionary {cluster_id: spike_ids}.

---

### phy.cluster.CorrelogramView

A view showing the autocorrelogram of the selected clusters, and all cross-correlograms
of cluster pairs.

**Constructor**


* `correlograms : function`
    Maps `(cluster_ids, bin_size, window_size)` to an `(n_clusters, n_clusters, n_bins) array`.


* `firing_rate : function`
    Maps `(cluster_ids, bin_size)` to an `(n_clusters, n_clusters) array`

---

#### CorrelogramView.attach


**`CorrelogramView.attach(self, gui)`**

Attach the view to the GUI.

---

#### CorrelogramView.close


**`CorrelogramView.close(self)`**

Close the view.

---

#### CorrelogramView.decrease


**`CorrelogramView.decrease(self)`**

Decrease the window size.

---

#### CorrelogramView.get_clusters_data


**`CorrelogramView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### CorrelogramView.increase


**`CorrelogramView.increase(self)`**

Increase the window size.

---

#### CorrelogramView.on_cluster


**`CorrelogramView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### CorrelogramView.on_mouse_wheel


**`CorrelogramView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### CorrelogramView.on_select


**`CorrelogramView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### CorrelogramView.on_select_threaded


**`CorrelogramView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### CorrelogramView.plot


**`CorrelogramView.plot(self, **kwargs)`**

Update the view with the current cluster selection.

---

#### CorrelogramView.reset_scaling


**`CorrelogramView.reset_scaling(self)`**

Reset the scaling to the default value.

---

#### CorrelogramView.screenshot


**`CorrelogramView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### CorrelogramView.set_bin


**`CorrelogramView.set_bin(self, bin_size)`**

Set the correlogram bin size (in milliseconds).

Example: `1`

---

#### CorrelogramView.set_refractory_period


**`CorrelogramView.set_refractory_period(self, value)`**

Set the refractory period (in milliseconds).

---

#### CorrelogramView.set_state


**`CorrelogramView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### CorrelogramView.set_window


**`CorrelogramView.set_window(self, window_size)`**

Set the correlogram window size (in milliseconds).

Example: `100`

---

#### CorrelogramView.show


**`CorrelogramView.show(self)`**

Show the underlying canvas.

---

#### CorrelogramView.toggle_auto_update


**`CorrelogramView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### CorrelogramView.toggle_labels


**`CorrelogramView.toggle_labels(self, checked)`**

Show or hide all labels.

---

#### CorrelogramView.toggle_normalization


**`CorrelogramView.toggle_normalization(self, checked)`**

Change the normalization of the correlograms.

---

#### CorrelogramView.update_status


**`CorrelogramView.update_status(self)`**



---

#### CorrelogramView.state


**`CorrelogramView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### CorrelogramView.status


**`CorrelogramView.status`**



---

### phy.cluster.FeatureView

This view displays a 4x4 subplot matrix with different projections of the principal
component features. This view keeps track of which channels are currently shown.

**Constructor**


* `features : function`
    Maps `(cluster_id, channel_ids=None, load_all=False)` to
    `Bunch(data, channel_ids, channel_labels, spike_ids , masks)`.
    * `data` is an `(n_spikes, n_channels, n_features)` array
    * `channel_ids` contains the channel ids of every row in `data`
    * `channel_labels` contains the channel labels of every row in `data`
    * `spike_ids` is a `(n_spikes,)` array
    * `masks` is an `(n_spikes, n_channels)` array

    This allows for a sparse format.


* `attributes : dict`
    Maps an attribute name to a 1D array with `n_spikes` numbers (for example, spike times).

---

#### FeatureView.attach


**`FeatureView.attach(self, gui)`**

Attach the view to the GUI.

---

#### FeatureView.clear_channels


**`FeatureView.clear_channels(self)`**

Reset the current channels.

---

#### FeatureView.close


**`FeatureView.close(self)`**

Close the view.

---

#### FeatureView.decrease


**`FeatureView.decrease(self)`**

Decrease the scaling parameter.

---

#### FeatureView.decrease_marker_size


**`FeatureView.decrease_marker_size(self)`**

Decrease the scaling parameter.

---

#### FeatureView.get_clusters_data


**`FeatureView.get_clusters_data(self, fixed_channels=None, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### FeatureView.increase


**`FeatureView.increase(self)`**

Increase the scaling parameter.

---

#### FeatureView.increase_marker_size


**`FeatureView.increase_marker_size(self)`**

Increase the scaling parameter.

---

#### FeatureView.on_cluster


**`FeatureView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### FeatureView.on_mouse_click


**`FeatureView.on_mouse_click(self, e)`**

Select a feature dimension by clicking on a box in the feature view.

---

#### FeatureView.on_mouse_wheel


**`FeatureView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### FeatureView.on_request_split


**`FeatureView.on_request_split(self, sender=None)`**

Return the spikes enclosed by the lasso.

---

#### FeatureView.on_select


**`FeatureView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### FeatureView.on_select_channel


**`FeatureView.on_select_channel(self, sender=None, channel_id=None, key=None, button=None)`**

Respond to the click on a channel from another view, and update the
relevant subplots.

---

#### FeatureView.on_select_threaded


**`FeatureView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### FeatureView.plot


**`FeatureView.plot(self, **kwargs)`**

Update the view with the selected clusters.

---

#### FeatureView.reset_marker_size


**`FeatureView.reset_marker_size(self)`**

Reset the scaling to the default value.

---

#### FeatureView.reset_scaling


**`FeatureView.reset_scaling(self)`**

Reset the scaling to the default value.

---

#### FeatureView.screenshot


**`FeatureView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### FeatureView.set_grid_dim


**`FeatureView.set_grid_dim(self, grid_dim)`**

Change the grid dim dynamically.

**Parameters**

* `grid_dim : array-like (2D)`
    `grid_dim[row, col]` is a string with two values separated by a comma. Each value
    is the relative channel id (0, 1, 2...) followed by the PC (A, B, C...). For example,
    `grid_dim[row, col] = 0B,1A`. Each value can also be an attribute name, for example
    `time`. For example, `grid_dim[row, col] = time,2C`.

---

#### FeatureView.set_state


**`FeatureView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### FeatureView.show


**`FeatureView.show(self)`**

Show the underlying canvas.

---

#### FeatureView.toggle_auto_update


**`FeatureView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### FeatureView.toggle_automatic_channel_selection


**`FeatureView.toggle_automatic_channel_selection(self, checked)`**

Toggle the automatic selection of channels when the cluster selection changes.

---

#### FeatureView.update_status


**`FeatureView.update_status(self)`**



---

#### FeatureView.marker_size


**`FeatureView.marker_size`**

Size of the spike markers, in pixels.

---

#### FeatureView.state


**`FeatureView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### FeatureView.status


**`FeatureView.status`**



---

### phy.cluster.FiringRateView

Histogram view showing the time-dependent firing rate.

---

#### FiringRateView.attach


**`FiringRateView.attach(self, gui)`**

Attach the view to the GUI.

---

#### FiringRateView.close


**`FiringRateView.close(self)`**

Close the view.

---

#### FiringRateView.decrease


**`FiringRateView.decrease(self)`**

Decrease the scaling parameter.

---

#### FiringRateView.get_clusters_data


**`FiringRateView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### FiringRateView.increase


**`FiringRateView.increase(self)`**

Increase the scaling parameter.

---

#### FiringRateView.on_cluster


**`FiringRateView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### FiringRateView.on_mouse_wheel


**`FiringRateView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### FiringRateView.on_select


**`FiringRateView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### FiringRateView.on_select_threaded


**`FiringRateView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### FiringRateView.plot


**`FiringRateView.plot(self, **kwargs)`**

Update the view with the selected clusters.

---

#### FiringRateView.reset_scaling


**`FiringRateView.reset_scaling(self)`**

Reset the scaling to the default value.

---

#### FiringRateView.screenshot


**`FiringRateView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### FiringRateView.set_bin_size


**`FiringRateView.set_bin_size(self, bin_size)`**

Set the bin size in the histogram.

---

#### FiringRateView.set_n_bins


**`FiringRateView.set_n_bins(self, n_bins)`**

Set the number of bins in the histogram.

---

#### FiringRateView.set_state


**`FiringRateView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### FiringRateView.set_x_max


**`FiringRateView.set_x_max(self, x_max)`**

Set the maximum value on the x axis for the histogram.

---

#### FiringRateView.set_x_min


**`FiringRateView.set_x_min(self, x_min)`**

Set the minimum value on the x axis for the histogram.

---

#### FiringRateView.show


**`FiringRateView.show(self)`**

Show the underlying canvas.

---

#### FiringRateView.toggle_auto_update


**`FiringRateView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### FiringRateView.update_status


**`FiringRateView.update_status(self)`**



---

#### FiringRateView.bin_size


**`FiringRateView.bin_size`**

Return the bin size (in seconds or milliseconds depending on `self.bin_unit`).

---

#### FiringRateView.state


**`FiringRateView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### FiringRateView.status


**`FiringRateView.status`**



---

### phy.cluster.HistogramView

This view displays a histogram for every selected cluster, along with a possible plot
and some text. To be overriden.

**Constructor**


* `cluster_stat : function`
    Maps `cluster_id` to `Bunch(data (1D array), plot (1D array), text)`.

---

#### HistogramView.attach


**`HistogramView.attach(self, gui)`**

Attach the view to the GUI.

---

#### HistogramView.close


**`HistogramView.close(self)`**

Close the view.

---

#### HistogramView.decrease


**`HistogramView.decrease(self)`**

Decrease the scaling parameter.

---

#### HistogramView.get_clusters_data


**`HistogramView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### HistogramView.increase


**`HistogramView.increase(self)`**

Increase the scaling parameter.

---

#### HistogramView.on_cluster


**`HistogramView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### HistogramView.on_mouse_wheel


**`HistogramView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### HistogramView.on_select


**`HistogramView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### HistogramView.on_select_threaded


**`HistogramView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### HistogramView.plot


**`HistogramView.plot(self, **kwargs)`**

Update the view with the selected clusters.

---

#### HistogramView.reset_scaling


**`HistogramView.reset_scaling(self)`**

Reset the scaling to the default value.

---

#### HistogramView.screenshot


**`HistogramView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### HistogramView.set_bin_size


**`HistogramView.set_bin_size(self, bin_size)`**

Set the bin size in the histogram.

---

#### HistogramView.set_n_bins


**`HistogramView.set_n_bins(self, n_bins)`**

Set the number of bins in the histogram.

---

#### HistogramView.set_state


**`HistogramView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### HistogramView.set_x_max


**`HistogramView.set_x_max(self, x_max)`**

Set the maximum value on the x axis for the histogram.

---

#### HistogramView.set_x_min


**`HistogramView.set_x_min(self, x_min)`**

Set the minimum value on the x axis for the histogram.

---

#### HistogramView.show


**`HistogramView.show(self)`**

Show the underlying canvas.

---

#### HistogramView.toggle_auto_update


**`HistogramView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### HistogramView.update_status


**`HistogramView.update_status(self)`**



---

#### HistogramView.bin_size


**`HistogramView.bin_size`**

Return the bin size (in seconds or milliseconds depending on `self.bin_unit`).

---

#### HistogramView.state


**`HistogramView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### HistogramView.status


**`HistogramView.status`**



---

### phy.cluster.ISIView

Histogram view showing the interspike intervals.

---

#### ISIView.attach


**`ISIView.attach(self, gui)`**

Attach the view to the GUI.

---

#### ISIView.close


**`ISIView.close(self)`**

Close the view.

---

#### ISIView.decrease


**`ISIView.decrease(self)`**

Decrease the scaling parameter.

---

#### ISIView.get_clusters_data


**`ISIView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### ISIView.increase


**`ISIView.increase(self)`**

Increase the scaling parameter.

---

#### ISIView.on_cluster


**`ISIView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### ISIView.on_mouse_wheel


**`ISIView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### ISIView.on_select


**`ISIView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### ISIView.on_select_threaded


**`ISIView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### ISIView.plot


**`ISIView.plot(self, **kwargs)`**

Update the view with the selected clusters.

---

#### ISIView.reset_scaling


**`ISIView.reset_scaling(self)`**

Reset the scaling to the default value.

---

#### ISIView.screenshot


**`ISIView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### ISIView.set_bin_size


**`ISIView.set_bin_size(self, bin_size)`**

Set the bin size in the histogram.

---

#### ISIView.set_n_bins


**`ISIView.set_n_bins(self, n_bins)`**

Set the number of bins in the histogram.

---

#### ISIView.set_state


**`ISIView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### ISIView.set_x_max


**`ISIView.set_x_max(self, x_max)`**

Set the maximum value on the x axis for the histogram.

---

#### ISIView.set_x_min


**`ISIView.set_x_min(self, x_min)`**

Set the minimum value on the x axis for the histogram.

---

#### ISIView.show


**`ISIView.show(self)`**

Show the underlying canvas.

---

#### ISIView.toggle_auto_update


**`ISIView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### ISIView.update_status


**`ISIView.update_status(self)`**



---

#### ISIView.bin_size


**`ISIView.bin_size`**

Return the bin size (in seconds or milliseconds depending on `self.bin_unit`).

---

#### ISIView.state


**`ISIView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### ISIView.status


**`ISIView.status`**



---

### phy.cluster.ManualClusteringView

Base class for clustering views.

Typical property objects:

- `self.canvas`: a `PlotCanvas` instance by default (can also be a `PlotCanvasMpl` instance).
- `self.default_shortcuts`: a dictionary with the default keyboard shortcuts for the view
- `self.shortcuts`: a dictionary with the actual keyboard shortcuts for the view (can be passed
  to the view's constructor).
- `self.state_attrs`: a tuple with all attributes that should be automatically saved in the
  view's global GUI state.
- `self.local_state_attrs`: like above, but for the local GUI state (dataset-dependent).

Events raised:

- `view_attached(view, gui)`: this is the event to connect to if you write a plugin that
  needs to modify a view.
- `is_busy(view)`
- `toggle_auto_update(view)`

---

#### ManualClusteringView.attach


**`ManualClusteringView.attach(self, gui)`**

Attach the view to the GUI.

Perform the following:

- Add the view to the GUI.
- Update the view's attribute from the GUI state
- Add the default view actions (auto_update, screenshot)
- Bind the on_select() method to the select event raised by the supervisor.

---

#### ManualClusteringView.close


**`ManualClusteringView.close(self)`**

Close the view.

---

#### ManualClusteringView.get_clusters_data


**`ManualClusteringView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### ManualClusteringView.on_cluster


**`ManualClusteringView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### ManualClusteringView.on_select


**`ManualClusteringView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### ManualClusteringView.on_select_threaded


**`ManualClusteringView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### ManualClusteringView.plot


**`ManualClusteringView.plot(self, **kwargs)`**

Update the view with the current cluster selection.

---

#### ManualClusteringView.screenshot


**`ManualClusteringView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### ManualClusteringView.set_state


**`ManualClusteringView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### ManualClusteringView.show


**`ManualClusteringView.show(self)`**

Show the underlying canvas.

---

#### ManualClusteringView.toggle_auto_update


**`ManualClusteringView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### ManualClusteringView.update_status


**`ManualClusteringView.update_status(self)`**



---

#### ManualClusteringView.state


**`ManualClusteringView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### ManualClusteringView.status


**`ManualClusteringView.status`**

To be overriden.

---

### phy.cluster.ProbeView

This view displays the positions of all channels on the probe, highlighting channels
where the selected clusters belong.

**Constructor**


* `positions : array-like`
    An `(n_channels, 2)` array with the channel positions

* `best_channels : function`
    Maps `cluster_id` to the list of the best_channel_ids.

* `channel_labels : list`
    List of channel label strings.

* `dead_channels : list`
    List of dead channel ids.

---

#### ProbeView.attach


**`ProbeView.attach(self, gui)`**

Attach the view to the GUI.

---

#### ProbeView.close


**`ProbeView.close(self)`**

Close the view.

---

#### ProbeView.get_clusters_data


**`ProbeView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### ProbeView.on_cluster


**`ProbeView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### ProbeView.on_select


**`ProbeView.on_select(self, cluster_ids=(), **kwargs)`**

Update the view with the selected clusters.

---

#### ProbeView.on_select_threaded


**`ProbeView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### ProbeView.plot


**`ProbeView.plot(self, **kwargs)`**

Update the view with the current cluster selection.

---

#### ProbeView.screenshot


**`ProbeView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### ProbeView.set_state


**`ProbeView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### ProbeView.show


**`ProbeView.show(self)`**

Show the underlying canvas.

---

#### ProbeView.toggle_auto_update


**`ProbeView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### ProbeView.toggle_show_labels


**`ProbeView.toggle_show_labels(self, checked)`**

Toggle the display of the channel ids.

---

#### ProbeView.update_status


**`ProbeView.update_status(self)`**



---

#### ProbeView.state


**`ProbeView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### ProbeView.status


**`ProbeView.status`**

To be overriden.

---

### phy.cluster.RasterView

This view shows a raster plot of all clusters.

**Constructor**


* `spike_times : array-like`
    An `(n_spikes,)` array with the spike times, in seconds.

* `spike_clusters : array-like`
    An `(n_spikes,)` array with the spike-cluster assignments.

* `cluster_ids : array-like`
    The list of all clusters to show initially.

---

#### RasterView.add_color_scheme


**`RasterView.add_color_scheme(self, fun=None, name=None, cluster_ids=None, colormap=None, categorical=None, logarithmic=None)`**

Add a color scheme to the view. Can be used as follows:

```python
@connect
def on_view_attached(gui, view):
    view.add_color_scheme(c.get_depth, name='depth', colormap='linear')
```

---

#### RasterView.attach


**`RasterView.attach(self, gui)`**

Attach the view to the GUI.

---

#### RasterView.close


**`RasterView.close(self)`**

Close the view.

---

#### RasterView.decrease_marker_size


**`RasterView.decrease_marker_size(self)`**

Decrease the scaling parameter.

---

#### RasterView.get_cluster_colors


**`RasterView.get_cluster_colors(self, cluster_ids, alpha=1.0)`**

Return the cluster colors depending on the currently-selected color scheme.

---

#### RasterView.get_clusters_data


**`RasterView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### RasterView.increase_marker_size


**`RasterView.increase_marker_size(self)`**

Increase the scaling parameter.

---

#### RasterView.next_color_scheme


**`RasterView.next_color_scheme(self)`**

Switch to the next color scheme.

---

#### RasterView.on_cluster


**`RasterView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### RasterView.on_mouse_click


**`RasterView.on_mouse_click(self, e)`**

Select a cluster by clicking in the raster plot.

---

#### RasterView.on_mouse_wheel


**`RasterView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### RasterView.on_select


**`RasterView.on_select(self, *args, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### RasterView.on_select_threaded


**`RasterView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### RasterView.plot


**`RasterView.plot(self, **kwargs)`**

Make the raster plot.

---

#### RasterView.previous_color_scheme


**`RasterView.previous_color_scheme(self)`**

Switch to the previous color scheme.

---

#### RasterView.reset_marker_size


**`RasterView.reset_marker_size(self)`**

Reset the scaling to the default value.

---

#### RasterView.screenshot


**`RasterView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### RasterView.set_cluster_ids


**`RasterView.set_cluster_ids(self, cluster_ids)`**

Set the shown clusters, which can be filtered and in any order (from top to bottom).

---

#### RasterView.set_spike_clusters


**`RasterView.set_spike_clusters(self, spike_clusters)`**

Set the spike clusters for all spikes.

---

#### RasterView.set_state


**`RasterView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### RasterView.show


**`RasterView.show(self)`**

Show the underlying canvas.

---

#### RasterView.toggle_auto_update


**`RasterView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### RasterView.update_cluster_sort


**`RasterView.update_cluster_sort(self, cluster_ids)`**

Update the order of all clusters.

---

#### RasterView.update_color


**`RasterView.update_color(self)`**

Update the color of the spikes, depending on the selected clusters.

---

#### RasterView.update_select_color


**`RasterView.update_select_color(self)`**

Update the cluster colors after the cluster selection changes.

---

#### RasterView.update_status


**`RasterView.update_status(self)`**



---

#### RasterView.zoom_to_time_range


**`RasterView.zoom_to_time_range(self, interval)`**

Zoom to a time interval.

---

#### RasterView.color_scheme


**`RasterView.color_scheme`**

Current color scheme.

---

#### RasterView.marker_size


**`RasterView.marker_size`**

Size of the spike markers, in pixels.

---

#### RasterView.state


**`RasterView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### RasterView.status


**`RasterView.status`**



---

### phy.cluster.ScatterView

This view displays a scatter plot for all selected clusters.

**Constructor**


* `coords : function`
    Maps `cluster_ids` to a list `[Bunch(x, y, spike_ids, data_bounds), ...]` for each cluster.

---

#### ScatterView.attach


**`ScatterView.attach(self, gui)`**



---

#### ScatterView.close


**`ScatterView.close(self)`**

Close the view.

---

#### ScatterView.decrease_marker_size


**`ScatterView.decrease_marker_size(self)`**

Decrease the scaling parameter.

---

#### ScatterView.get_clusters_data


**`ScatterView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

---

#### ScatterView.increase_marker_size


**`ScatterView.increase_marker_size(self)`**

Increase the scaling parameter.

---

#### ScatterView.on_cluster


**`ScatterView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### ScatterView.on_mouse_wheel


**`ScatterView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### ScatterView.on_request_split


**`ScatterView.on_request_split(self, sender=None)`**

Return the spikes enclosed by the lasso.

---

#### ScatterView.on_select


**`ScatterView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### ScatterView.on_select_threaded


**`ScatterView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### ScatterView.plot


**`ScatterView.plot(self, **kwargs)`**

Update the view with the current cluster selection.

---

#### ScatterView.reset_marker_size


**`ScatterView.reset_marker_size(self)`**

Reset the scaling to the default value.

---

#### ScatterView.screenshot


**`ScatterView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### ScatterView.set_state


**`ScatterView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### ScatterView.show


**`ScatterView.show(self)`**

Show the underlying canvas.

---

#### ScatterView.toggle_auto_update


**`ScatterView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### ScatterView.update_status


**`ScatterView.update_status(self)`**



---

#### ScatterView.marker_size


**`ScatterView.marker_size`**

Size of the spike markers, in pixels.

---

#### ScatterView.state


**`ScatterView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### ScatterView.status


**`ScatterView.status`**

To be overriden.

---

### phy.cluster.SimilarityView

Display a table of clusters with metrics and labels as columns, and an additional
similarity column.

This view displays clusters similar to the clusters currently selected
in the cluster view.

**Events**

* request_similar_clusters(cluster_id)

---

#### SimilarityView.add


**`SimilarityView.add(self, objects)`**

Add objects object to the table.

---

#### SimilarityView.build


**`SimilarityView.build(self, callback=None)`**

Rebuild the HTML code of the widget.

---

#### SimilarityView.change


**`SimilarityView.change(self, objects)`**

Change some objects.

---

#### SimilarityView.eval_js


**`SimilarityView.eval_js(self, expr, callback=None)`**

Evaluate a Javascript expression.

The `table` Javascript variable can be used to interact with the underlying Javascript
table.

The table has sortable columns, a filter text box, support for single and multi selection
of rows. Rows can be skippable (used for ignored clusters in phy).

The table can raise Javascript events that are relayed to Python. Objects are
transparently serialized and deserialized in JSON. Basic types (numbers, strings, lists)
are transparently converted between Python and Javascript.

**Parameters**


* `expr : str`
    A Javascript expression.

* `callback : function`
    A Python function that is called once the Javascript expression has been
    evaluated. It takes as input the output of the Javascript expression.

---

#### SimilarityView.filter


**`SimilarityView.filter(self, text='')`**

Filter the view with a Javascript expression.

---

#### SimilarityView.first


**`SimilarityView.first(self, callback=None)`**

Select the first item.

---

#### SimilarityView.get


**`SimilarityView.get(self, id, callback=None)`**

Get the object given its id.

---

#### SimilarityView.get_current_sort


**`SimilarityView.get_current_sort(self, callback=None)`**

Get the current sort as a tuple `(name, dir)`.

---

#### SimilarityView.get_ids


**`SimilarityView.get_ids(self, callback=None)`**

Get the list of ids.

---

#### SimilarityView.get_next_id


**`SimilarityView.get_next_id(self, callback=None)`**

Get the next non-skipped row id.

---

#### SimilarityView.get_previous_id


**`SimilarityView.get_previous_id(self, callback=None)`**

Get the previous non-skipped row id.

---

#### SimilarityView.get_selected


**`SimilarityView.get_selected(self, callback=None)`**

Get the currently selected rows.

---

#### SimilarityView.is_ready


**`SimilarityView.is_ready(self)`**

Whether the widget has been fully loaded.

---

#### SimilarityView.last


**`SimilarityView.last(self, callback=None)`**

Select the last item.

---

#### SimilarityView.next


**`SimilarityView.next(self, callback=None)`**

Select the next non-skipped row.

---

#### SimilarityView.previous


**`SimilarityView.previous(self, callback=None)`**

Select the previous non-skipped row.

---

#### SimilarityView.remove


**`SimilarityView.remove(self, ids)`**

Remove some objects from their ids.

---

#### SimilarityView.remove_all


**`SimilarityView.remove_all(self)`**

Remove all rows in the table.

---

#### SimilarityView.remove_all_and_add


**`SimilarityView.remove_all_and_add(self, objects)`**

Remove all rows in the table and add new objects.

---

#### SimilarityView.reset


**`SimilarityView.reset(self, cluster_ids)`**

Recreate the similarity view, given the selected clusters in the cluster view.

---

#### SimilarityView.scroll_to


**`SimilarityView.scroll_to(self, id)`**

Scroll until a given row is visible.

---

#### SimilarityView.select


**`SimilarityView.select(self, ids, callback=None, **kwargs)`**

Select some rows in the table from Python.

This function calls `table.select()` in Javascript, which raises a Javascript event
relayed to Python. This sequence of actions is the same when the user selects
rows directly in the HTML view.

---

#### SimilarityView.set_busy


**`SimilarityView.set_busy(self, busy)`**

Set the busy state of the GUI.

---

#### SimilarityView.set_html


**`SimilarityView.set_html(self, html, callback=None)`**

Set the HTML code.

---

#### SimilarityView.set_selected_index_offset


**`SimilarityView.set_selected_index_offset(self, n)`**

Set the index of the selected cluster, used for correct coloring in the similarity
view.

---

#### SimilarityView.set_state


**`SimilarityView.set_state(self, state)`**

Set the cluster view state, with a specified sort.

---

#### SimilarityView.sort_by


**`SimilarityView.sort_by(self, name, sort_dir='asc')`**

Sort by a given variable.

---

#### SimilarityView.view_source


**`SimilarityView.view_source(self, callback=None)`**

View the HTML source of the widget.

---

#### SimilarityView.debouncer


**`SimilarityView.debouncer`**

Widget debouncer.

---

#### SimilarityView.state


**`SimilarityView.state`**

Return the cluster view state, with the current sort and selection.

---

### phy.cluster.Supervisor

Component that brings manual clustering facilities to a GUI:

* `Clustering` instance: merge, split, undo, redo.
* `ClusterMeta` instance: change cluster metadata (e.g. group).
* Cluster selection.
* Many manual clustering-related actions, snippets, shortcuts, etc.
* Two HTML tables : `ClusterView` and `SimilarityView`.

**Constructor**


* `spike_clusters : array-like`
    Spike-clusters assignments.

* `cluster_groups : dict`
    Maps a cluster id to a group name (noise, mea, good, None for unsorted).

* `cluster_metrics : dict`
    Maps a metric name to a function `cluster_id => value`

* `similarity : function`
    Maps a cluster id to a list of pairs `[(similar_cluster_id, similarity), ...]`

* `new_cluster_id : function`
    Function that takes no argument and returns a brand new cluster id (smallest cluster id
    not used in the cache).

* `sort : 2-tuple`
    Initial sort as a pair `(column_name, order)` where `order` is either `asc` or `desc`

* `context : Context`
    Handles the cache.

**Events**

When this component is attached to a GUI, the following events are emitted:

* `select(cluster_ids)`
    When clusters are selected in the cluster view or similarity view.
* `cluster(up)`
    When a clustering action occurs, changing the spike clusters assignment of the cluster
    metadata.
* `attach_gui(gui)`
    When the Supervisor instance is attached to the GUI.
* `request_split()`
    When the user requests to split (typically, a lasso has been drawn before).
* `save_clustering(spike_clusters, cluster_groups, *cluster_labels)`
    When the user wants to save the spike cluster assignments and the cluster metadata.

---

#### Supervisor.attach


**`Supervisor.attach(self, gui)`**

Attach to the GUI.

---

#### Supervisor.block


**`Supervisor.block(self)`**

Block until there are no pending actions.

Only used in the automated testing suite.

---

#### Supervisor.clear_filter


**`Supervisor.clear_filter(self)`**



---

#### Supervisor.filter


**`Supervisor.filter(self, text)`**

Filter the clusters using a Javascript expression on the column names.

---

#### Supervisor.first


**`Supervisor.first(self, callback=None)`**

Select the first cluster in the cluster view.

---

#### Supervisor.get_cluster_info


**`Supervisor.get_cluster_info(self, cluster_id, exclude=())`**

Return the data associated to a given cluster.

---

#### Supervisor.get_labels


**`Supervisor.get_labels(self, field)`**

Return the labels of all clusters, for a given label name.

---

#### Supervisor.is_dirty


**`Supervisor.is_dirty(self)`**

Return whether there are any pending changes.

---

#### Supervisor.label


**`Supervisor.label(self, name, value, cluster_ids=None)`**

Assign a label to some clusters.

---

#### Supervisor.last


**`Supervisor.last(self, callback=None)`**

Select the last cluster in the cluster view.

---

#### Supervisor.merge


**`Supervisor.merge(self, cluster_ids=None, to=None)`**

Merge the selected clusters.

---

#### Supervisor.move


**`Supervisor.move(self, group, which)`**

Assign a cluster group to some clusters.

---

#### Supervisor.n_spikes


**`Supervisor.n_spikes(self, cluster_id)`**

Number of spikes in a given cluster.

---

#### Supervisor.next


**`Supervisor.next(self, callback=None)`**

Select the next cluster in the similarity view.

---

#### Supervisor.next_best


**`Supervisor.next_best(self, callback=None)`**

Select the next best cluster in the cluster view.

---

#### Supervisor.previous


**`Supervisor.previous(self, callback=None)`**

Select the previous cluster in the similarity view.

---

#### Supervisor.previous_best


**`Supervisor.previous_best(self, callback=None)`**

Select the previous best cluster in the cluster view.

---

#### Supervisor.redo


**`Supervisor.redo(self)`**

Undo the last undone action.

---

#### Supervisor.reset_wizard


**`Supervisor.reset_wizard(self, callback=None)`**

Reset the wizard.

---

#### Supervisor.save


**`Supervisor.save(self)`**

Save the manual clustering back to disk.

This method emits the `save_clustering(spike_clusters, groups, *labels)` event.
It is up to the caller to react to this event and save the data to disk.

---

#### Supervisor.select


**`Supervisor.select(self, *cluster_ids, callback=None)`**

Select a list of clusters.

---

#### Supervisor.sort


**`Supervisor.sort(self, column, sort_dir='desc')`**

Sort the cluster view by a given column, in a given order (asc or desc).

---

#### Supervisor.split


**`Supervisor.split(self, spike_ids=None, spike_clusters_rel=0)`**

Make a new cluster out of the specified spikes.

---

#### Supervisor.undo


**`Supervisor.undo(self)`**

Undo the last action.

---

#### Supervisor.unselect_similar


**`Supervisor.unselect_similar(self, callback=None)`**

Select only the clusters in the cluster view.

---

#### Supervisor.cluster_info


**`Supervisor.cluster_info`**

The cluster view table as a list of per-cluster dictionaries.

---

#### Supervisor.fields


**`Supervisor.fields`**

List of all cluster label names.

---

#### Supervisor.selected


**`Supervisor.selected`**

Selected clusters in the cluster and similarity views.

---

#### Supervisor.selected_clusters


**`Supervisor.selected_clusters`**

Selected clusters in the cluster view only.

---

#### Supervisor.selected_similar


**`Supervisor.selected_similar`**

Selected clusters in the similarity view only.

---

#### Supervisor.shown_cluster_ids


**`Supervisor.shown_cluster_ids`**

The sorted list of cluster ids as they are currently shown in the cluster view.

---

#### Supervisor.state


**`Supervisor.state`**

GUI state, with the cluster view and similarity view states.

---

### phy.cluster.TemplateView

This view shows all template waveforms of all clusters in a large grid of shape
`(n_channels, n_clusters)`.

**Constructor**


* `templates : function`
    Maps `cluster_ids` to a list of `[Bunch(template, channel_ids)]` where `template` is
    an `(n_samples, n_channels)` array, and `channel_ids` specifies the channels of the
    `template` array (sparse format).

* `channel_ids : array-like`
    The list of all channel ids.

* `channel_labels : list`
    Labels of all shown channels. By default, this is just the channel ids.

* `cluster_ids : array-like`
    The list of all clusters to show initially.

---

#### TemplateView.add_color_scheme


**`TemplateView.add_color_scheme(self, fun=None, name=None, cluster_ids=None, colormap=None, categorical=None, logarithmic=None)`**

Add a color scheme to the view. Can be used as follows:

```python
@connect
def on_view_attached(gui, view):
    view.add_color_scheme(c.get_depth, name='depth', colormap='linear')
```

---

#### TemplateView.attach


**`TemplateView.attach(self, gui)`**



---

#### TemplateView.close


**`TemplateView.close(self)`**

Close the view.

---

#### TemplateView.decrease


**`TemplateView.decrease(self)`**

Decrease the scaling parameter.

---

#### TemplateView.get_cluster_colors


**`TemplateView.get_cluster_colors(self, cluster_ids, alpha=1.0)`**

Return the cluster colors depending on the currently-selected color scheme.

---

#### TemplateView.get_clusters_data


**`TemplateView.get_clusters_data(self, load_all=None)`**

Return all templates data.

---

#### TemplateView.increase


**`TemplateView.increase(self)`**

Increase the scaling parameter.

---

#### TemplateView.next_color_scheme


**`TemplateView.next_color_scheme(self)`**

Switch to the next color scheme.

---

#### TemplateView.on_cluster


**`TemplateView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### TemplateView.on_mouse_click


**`TemplateView.on_mouse_click(self, e)`**

Select a cluster by clicking on its template waveform.

---

#### TemplateView.on_mouse_wheel


**`TemplateView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### TemplateView.on_select


**`TemplateView.on_select(self, *args, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### TemplateView.on_select_threaded


**`TemplateView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### TemplateView.plot


**`TemplateView.plot(self, **kwargs)`**

Make the template plot.

---

#### TemplateView.previous_color_scheme


**`TemplateView.previous_color_scheme(self)`**

Switch to the previous color scheme.

---

#### TemplateView.reset_scaling


**`TemplateView.reset_scaling(self)`**

Reset the scaling to the default value.

---

#### TemplateView.screenshot


**`TemplateView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### TemplateView.set_cluster_ids


**`TemplateView.set_cluster_ids(self, cluster_ids)`**

Update the cluster ids when their identity or order has changed.

---

#### TemplateView.set_spike_clusters


**`TemplateView.set_spike_clusters(self, spike_clusters)`**



---

#### TemplateView.set_state


**`TemplateView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### TemplateView.show


**`TemplateView.show(self)`**

Show the underlying canvas.

---

#### TemplateView.toggle_auto_update


**`TemplateView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### TemplateView.update_cluster_sort


**`TemplateView.update_cluster_sort(self, cluster_ids)`**

Update the order of the clusters.

---

#### TemplateView.update_color


**`TemplateView.update_color(self)`**

Update the color of the clusters, taking the selected clusters into account.

---

#### TemplateView.update_select_color


**`TemplateView.update_select_color(self)`**

Update the cluster colors after the cluster selection changes.

---

#### TemplateView.update_status


**`TemplateView.update_status(self)`**



---

#### TemplateView.color_scheme


**`TemplateView.color_scheme`**

Current color scheme.

---

#### TemplateView.scaling


**`TemplateView.scaling`**

Return the grid scaling.

---

#### TemplateView.state


**`TemplateView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### TemplateView.status


**`TemplateView.status`**



---

### phy.cluster.TraceImageView

This view shows the raw traces as an image

**Constructor**


* `traces : function`
    Maps a time interval `(t0, t1)` to a `Bunch(data, color, waveforms)` where
    * `data` is an `(n_samples, n_channels)` array
    * `waveforms` is a list of bunchs with the following attributes:
        * `data`
        * `color`
        * `channel_ids`
        * `start_time`


* `sample_rate : float`

* `duration : float`

* `n_channels : int`

* `channel_positions : array-like`
    Positions of the channels, used for displaying the channels in the right y order

* `channel_labels : list`
    Labels of all shown channels. By default, this is just the channel ids.

---

#### TraceImageView.add_color_scheme


**`TraceImageView.add_color_scheme(self, fun=None, name=None, cluster_ids=None, colormap=None, categorical=None, logarithmic=None)`**

Add a color scheme to the view. Can be used as follows:

```python
@connect
def on_view_attached(gui, view):
    view.add_color_scheme(c.get_depth, name='depth', colormap='linear')
```

---

#### TraceImageView.attach


**`TraceImageView.attach(self, gui)`**

Attach the view to the GUI.

---

#### TraceImageView.close


**`TraceImageView.close(self)`**

Close the view.

---

#### TraceImageView.decrease


**`TraceImageView.decrease(self)`**

Decrease the scaling parameter.

---

#### TraceImageView.get_cluster_colors


**`TraceImageView.get_cluster_colors(self, cluster_ids, alpha=1.0)`**

Return the cluster colors depending on the currently-selected color scheme.

---

#### TraceImageView.get_clusters_data


**`TraceImageView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### TraceImageView.go_left


**`TraceImageView.go_left(self)`**

Go to left.

---

#### TraceImageView.go_right


**`TraceImageView.go_right(self)`**

Go to right.

---

#### TraceImageView.go_to


**`TraceImageView.go_to(self, time)`**

Go to a specific time (in seconds).

---

#### TraceImageView.go_to_end


**`TraceImageView.go_to_end(self)`**

Go to end of the recording.

---

#### TraceImageView.go_to_next_spike


**`TraceImageView.go_to_next_spike(self)`**

Jump to the next spike from the first selected cluster.

---

#### TraceImageView.go_to_previous_spike


**`TraceImageView.go_to_previous_spike(self)`**

Jump to the previous spike from the first selected cluster.

---

#### TraceImageView.go_to_start


**`TraceImageView.go_to_start(self)`**

Go to the start of the recording.

---

#### TraceImageView.increase


**`TraceImageView.increase(self)`**

Increase the scaling parameter.

---

#### TraceImageView.jump_left


**`TraceImageView.jump_left(self)`**

Jump to left.

---

#### TraceImageView.jump_right


**`TraceImageView.jump_right(self)`**

Jump to right.

---

#### TraceImageView.narrow


**`TraceImageView.narrow(self)`**

Decrease the interval size.

---

#### TraceImageView.next_color_scheme


**`TraceImageView.next_color_scheme(self)`**

Switch to the next color scheme.

---

#### TraceImageView.on_cluster


**`TraceImageView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### TraceImageView.on_mouse_click


**`TraceImageView.on_mouse_click(self, e)`**

Select a cluster by clicking on a spike.

---

#### TraceImageView.on_mouse_wheel


**`TraceImageView.on_mouse_wheel(self, e)`**

Scroll through the data with alt+wheel.

---

#### TraceImageView.on_select


**`TraceImageView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### TraceImageView.on_select_threaded


**`TraceImageView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### TraceImageView.plot


**`TraceImageView.plot(self, update_traces=True, **kwargs)`**

Update the view with the current cluster selection.

---

#### TraceImageView.previous_color_scheme


**`TraceImageView.previous_color_scheme(self)`**

Switch to the previous color scheme.

---

#### TraceImageView.reset_scaling


**`TraceImageView.reset_scaling(self)`**

Reset the scaling to the default value.

---

#### TraceImageView.screenshot


**`TraceImageView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### TraceImageView.set_interval


**`TraceImageView.set_interval(self, interval=None)`**

Display the traces and spikes in a given interval.

---

#### TraceImageView.set_state


**`TraceImageView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### TraceImageView.shift


**`TraceImageView.shift(self, delay)`**

Shift the interval by a given delay (in seconds).

---

#### TraceImageView.show


**`TraceImageView.show(self)`**

Show the underlying canvas.

---

#### TraceImageView.switch_origin


**`TraceImageView.switch_origin(self)`**

Switch between top and bottom origin for the channels.

---

#### TraceImageView.toggle_auto_scale


**`TraceImageView.toggle_auto_scale(self, checked)`**

Toggle automatic scaling of the traces.

---

#### TraceImageView.toggle_auto_update


**`TraceImageView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### TraceImageView.toggle_highlighted_spikes


**`TraceImageView.toggle_highlighted_spikes(self, checked)`**

Toggle between showing all spikes or selected spikes.

---

#### TraceImageView.toggle_show_labels


**`TraceImageView.toggle_show_labels(self, checked)`**

Toggle the display of the channel ids.

---

#### TraceImageView.update_color


**`TraceImageView.update_color(self)`**

Update the view when the color scheme changes.

---

#### TraceImageView.update_select_color


**`TraceImageView.update_select_color(self)`**

Update the cluster colors after the cluster selection changes.

---

#### TraceImageView.update_status


**`TraceImageView.update_status(self)`**



---

#### TraceImageView.widen


**`TraceImageView.widen(self)`**

Increase the interval size.

---

#### TraceImageView.color_scheme


**`TraceImageView.color_scheme`**

Current color scheme.

---

#### TraceImageView.half_duration


**`TraceImageView.half_duration`**

Half of the duration of the current interval.

---

#### TraceImageView.interval


**`TraceImageView.interval`**

Interval as `(tmin, tmax)`.

---

#### TraceImageView.origin


**`TraceImageView.origin`**

Whether to show the channels from top to bottom (`top` option, the default), or from
bottom to top (`bottom`).

---

#### TraceImageView.scaling


**`TraceImageView.scaling`**

Scaling of the colormap vrange.

---

#### TraceImageView.stacked


**`TraceImageView.stacked`**



---

#### TraceImageView.state


**`TraceImageView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### TraceImageView.status


**`TraceImageView.status`**



---

#### TraceImageView.time


**`TraceImageView.time`**

Time at the center of the window.

---

### phy.cluster.TraceView

This view shows the raw traces along with spike waveforms.

**Constructor**


* `traces : function`
    Maps a time interval `(t0, t1)` to a `Bunch(data, color, waveforms)` where
    * `data` is an `(n_samples, n_channels)` array
    * `waveforms` is a list of bunchs with the following attributes:
        * `data`
        * `color`
        * `channel_ids`
        * `start_time`
        * `spike_id`
        * `spike_cluster`


* `spike_times : function`
    Teturns the list of relevant spike times.

* `sample_rate : float`

* `duration : float`

* `n_channels : int`

* `channel_positions : array-like`
    Positions of the channels, used for displaying the channels in the right y order

* `channel_labels : list`
    Labels of all shown channels. By default, this is just the channel ids.

---

#### TraceView.add_color_scheme


**`TraceView.add_color_scheme(self, fun=None, name=None, cluster_ids=None, colormap=None, categorical=None, logarithmic=None)`**

Add a color scheme to the view. Can be used as follows:

```python
@connect
def on_view_attached(gui, view):
    view.add_color_scheme(c.get_depth, name='depth', colormap='linear')
```

---

#### TraceView.attach


**`TraceView.attach(self, gui)`**

Attach the view to the GUI.

---

#### TraceView.close


**`TraceView.close(self)`**

Close the view.

---

#### TraceView.decrease


**`TraceView.decrease(self)`**

Decrease the scaling parameter.

---

#### TraceView.get_cluster_colors


**`TraceView.get_cluster_colors(self, cluster_ids, alpha=1.0)`**

Return the cluster colors depending on the currently-selected color scheme.

---

#### TraceView.get_clusters_data


**`TraceView.get_clusters_data(self, load_all=None)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### TraceView.go_left


**`TraceView.go_left(self)`**

Go to left.

---

#### TraceView.go_right


**`TraceView.go_right(self)`**

Go to right.

---

#### TraceView.go_to


**`TraceView.go_to(self, time)`**

Go to a specific time (in seconds).

---

#### TraceView.go_to_end


**`TraceView.go_to_end(self)`**

Go to end of the recording.

---

#### TraceView.go_to_next_spike


**`TraceView.go_to_next_spike(self)`**

Jump to the next spike from the first selected cluster.

---

#### TraceView.go_to_previous_spike


**`TraceView.go_to_previous_spike(self)`**

Jump to the previous spike from the first selected cluster.

---

#### TraceView.go_to_start


**`TraceView.go_to_start(self)`**

Go to the start of the recording.

---

#### TraceView.increase


**`TraceView.increase(self)`**

Increase the scaling parameter.

---

#### TraceView.jump_left


**`TraceView.jump_left(self)`**

Jump to left.

---

#### TraceView.jump_right


**`TraceView.jump_right(self)`**

Jump to right.

---

#### TraceView.narrow


**`TraceView.narrow(self)`**

Decrease the interval size.

---

#### TraceView.next_color_scheme


**`TraceView.next_color_scheme(self)`**

Switch to the next color scheme.

---

#### TraceView.on_cluster


**`TraceView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### TraceView.on_mouse_click


**`TraceView.on_mouse_click(self, e)`**

Select a cluster by clicking on a spike.

---

#### TraceView.on_mouse_wheel


**`TraceView.on_mouse_wheel(self, e)`**

Scroll through the data with alt+wheel.

---

#### TraceView.on_select


**`TraceView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### TraceView.on_select_threaded


**`TraceView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### TraceView.plot


**`TraceView.plot(self, update_traces=True, update_waveforms=True)`**

Update the view with the current cluster selection.

---

#### TraceView.previous_color_scheme


**`TraceView.previous_color_scheme(self)`**

Switch to the previous color scheme.

---

#### TraceView.reset_scaling


**`TraceView.reset_scaling(self)`**

Reset the scaling to the default value.

---

#### TraceView.screenshot


**`TraceView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### TraceView.set_interval


**`TraceView.set_interval(self, interval=None)`**

Display the traces and spikes in a given interval.

---

#### TraceView.set_state


**`TraceView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### TraceView.shift


**`TraceView.shift(self, delay)`**

Shift the interval by a given delay (in seconds).

---

#### TraceView.show


**`TraceView.show(self)`**

Show the underlying canvas.

---

#### TraceView.switch_origin


**`TraceView.switch_origin(self)`**

Switch between top and bottom origin for the channels.

---

#### TraceView.toggle_auto_scale


**`TraceView.toggle_auto_scale(self, checked)`**

Toggle automatic scaling of the traces.

---

#### TraceView.toggle_auto_update


**`TraceView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### TraceView.toggle_highlighted_spikes


**`TraceView.toggle_highlighted_spikes(self, checked)`**

Toggle between showing all spikes or selected spikes.

---

#### TraceView.toggle_show_labels


**`TraceView.toggle_show_labels(self, checked)`**

Toggle the display of the channel ids.

---

#### TraceView.update_color


**`TraceView.update_color(self)`**

Update the view when the color scheme changes.

---

#### TraceView.update_select_color


**`TraceView.update_select_color(self)`**

Update the cluster colors after the cluster selection changes.

---

#### TraceView.update_status


**`TraceView.update_status(self)`**



---

#### TraceView.widen


**`TraceView.widen(self)`**

Increase the interval size.

---

#### TraceView.color_scheme


**`TraceView.color_scheme`**

Current color scheme.

---

#### TraceView.half_duration


**`TraceView.half_duration`**

Half of the duration of the current interval.

---

#### TraceView.interval


**`TraceView.interval`**

Interval as `(tmin, tmax)`.

---

#### TraceView.origin


**`TraceView.origin`**

Whether to show the channels from top to bottom (`top` option, the default), or from
bottom to top (`bottom`).

---

#### TraceView.scaling


**`TraceView.scaling`**

Scaling of the channel boxes.

---

#### TraceView.stacked


**`TraceView.stacked`**



---

#### TraceView.state


**`TraceView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### TraceView.status


**`TraceView.status`**



---

#### TraceView.time


**`TraceView.time`**

Time at the center of the window.

---

### phy.cluster.UpdateInfo

Object created every time the dataset is modified via a clustering or cluster metadata
action. It is passed to event callbacks that react to these changes. Derive from Bunch.

**Parameters**


* `description : str`
    Information about the update: merge, assign, or metadata_xxx for metadata changes

* `history : str`
    undo, redo, or None

* `spike_ids : array-like`
    All spike ids that were affected by the clustering action.

* `added : list`
    List of new cluster ids.

* `deleted : list`
    List of cluster ids that were deleted during the action. There are no modified clusters:
    every change triggers the deletion of and addition of clusters.

* `descendants : list`
    List of pairs (old_cluster_id, new_cluster_id), used to track the history of
    the clusters.

* `metadata_changed : list`
    List of cluster ids that had a change of metadata.

* `metadata_value : str`
    The new metadata value for the affected change.

* `undo_state : Bunch`
    Returned during an undo, it contains information about the undone action. This is used
    when redoing the undone action.

---

#### UpdateInfo.copy


**`UpdateInfo.copy(self)`**

Return a new Bunch instance which is a copy of the current Bunch instance.

---

### phy.cluster.WaveformView

This view shows the waveforms of the selected clusters, on relevant channels,
following the probe geometry.

**Constructor**


* `waveforms : dict of functions`
    Every function maps a cluster id to a Bunch with the following attributes:

    * `data` : a 3D array `(n_spikes, n_samples, n_channels_loc)`
    * `channel_ids` : the channel ids corresponding to the third dimension in `data`
    * `channel_labels` : a list of channel labels for every channel in `channel_ids`
    * `channel_positions` : a 2D array with the coordinates of the channels on the probe
    * `masks` : a 2D array `(n_spikes, n_channels)` with the waveforms masks
    * `alpha` : the alpha transparency channel

    The keys of the dictionary are called **waveform types**. The `next_waveforms_type`
    action cycles through all available waveform types. The key `waveforms` is mandatory.

* `waveforms_type : str`
    Default key of the waveforms dictionary to plot initially.

---

#### WaveformView.attach


**`WaveformView.attach(self, gui)`**

Attach the view to the GUI.

---

#### WaveformView.close


**`WaveformView.close(self)`**

Close the view.

---

#### WaveformView.decrease


**`WaveformView.decrease(self)`**

Decrease the scaling parameter.

---

#### WaveformView.extend_horizontally


**`WaveformView.extend_horizontally(self)`**

Increase the horizontal scaling of the probe.

---

#### WaveformView.extend_vertically


**`WaveformView.extend_vertically(self)`**

Increase the vertical scaling of the waveforms.

---

#### WaveformView.get_clusters_data


**`WaveformView.get_clusters_data(self)`**

Return a list of Bunch instances, with attributes pos and spike_ids.

To override.

---

#### WaveformView.increase


**`WaveformView.increase(self)`**

Increase the scaling parameter.

---

#### WaveformView.narrow


**`WaveformView.narrow(self)`**

Decrease the horizontal scaling of the waveforms.

---

#### WaveformView.next_waveforms_type


**`WaveformView.next_waveforms_type(self)`**

Switch to the next waveforms type.

---

#### WaveformView.on_cluster


**`WaveformView.on_cluster(self, up)`**

Callback function when a clustering action occurs. May be overriden.

Note: this method is called *before* on_select() so as to give a chance to the view
to update itself before the selection of the new clusters.

This method is mostly only useful to views that show all clusters and not just the
selected clusters (template view, raster view).

---

#### WaveformView.on_mouse_click


**`WaveformView.on_mouse_click(self, e)`**

Select a channel by clicking on a box in the waveform view.

---

#### WaveformView.on_mouse_wheel


**`WaveformView.on_mouse_wheel(self, e)`**

Change the scaling with the wheel.

---

#### WaveformView.on_select


**`WaveformView.on_select(self, cluster_ids=None, **kwargs)`**

Callback function when clusters are selected. May be overriden.

---

#### WaveformView.on_select_threaded


**`WaveformView.on_select_threaded(self, sender, cluster_ids, gui=None, **kwargs)`**



---

#### WaveformView.plot


**`WaveformView.plot(self, **kwargs)`**

Update the view with the current cluster selection.

---

#### WaveformView.previous_waveforms_type


**`WaveformView.previous_waveforms_type(self)`**

Switch to the previous waveforms type.

---

#### WaveformView.reset_scaling


**`WaveformView.reset_scaling(self)`**

Reset the scaling to the default value.

---

#### WaveformView.screenshot


**`WaveformView.screenshot(self, dir=None)`**

Save a PNG screenshot of the view into a given directory. By default, the screenshots
are saved in `~/.phy/screenshots/`.

---

#### WaveformView.set_state


**`WaveformView.set_state(self, state)`**

Set the view state.

The passed object is the persisted `self.state` bunch.

May be overriden.

---

#### WaveformView.show


**`WaveformView.show(self)`**

Show the underlying canvas.

---

#### WaveformView.shrink_horizontally


**`WaveformView.shrink_horizontally(self)`**

Decrease the horizontal scaling of the waveforms.

---

#### WaveformView.shrink_vertically


**`WaveformView.shrink_vertically(self)`**

Decrease the vertical scaling of the waveforms.

---

#### WaveformView.toggle_auto_update


**`WaveformView.toggle_auto_update(self, checked)`**

When on, the view is automatically updated when the cluster selection changes.

---

#### WaveformView.toggle_mean_waveforms


**`WaveformView.toggle_mean_waveforms(self, checked)`**

Switch to the `mean_waveforms` type, if it is available.

---

#### WaveformView.toggle_show_labels


**`WaveformView.toggle_show_labels(self, checked)`**

Whether to show the channel ids or not.

---

#### WaveformView.toggle_waveform_overlap


**`WaveformView.toggle_waveform_overlap(self, checked)`**

Toggle the overlap of the waveforms.

---

#### WaveformView.update_status


**`WaveformView.update_status(self)`**



---

#### WaveformView.widen


**`WaveformView.widen(self)`**

Increase the horizontal scaling of the waveforms.

---

#### WaveformView.box_scaling


**`WaveformView.box_scaling`**



---

#### WaveformView.boxed


**`WaveformView.boxed`**

Layout instance.

---

#### WaveformView.overlap


**`WaveformView.overlap`**

Whether to overlap the waveforms belonging to different clusters.

---

#### WaveformView.probe_scaling


**`WaveformView.probe_scaling`**



---

#### WaveformView.state


**`WaveformView.state`**

View state, a Bunch instance automatically persisted in the GUI state when the
GUI is closed. To be overriden.

---

#### WaveformView.status


**`WaveformView.status`**



---

#### WaveformView.waveforms_type


**`WaveformView.waveforms_type`**



---

## phy.apps

CLI tool.

---

#### phy.apps.add_default_handler


**`phy.apps.add_default_handler(level='INFO', logger=<Logger phylib (DEBUG)>)`**



---

#### phy.apps.capture_exceptions


**`phy.apps.capture_exceptions()`**

Log exceptions instead of crashing the GUI, and display an error dialog on errors.

---

#### phy.apps.contextmanager


**`phy.apps.contextmanager(func)`**

@contextmanager decorator.

Typical usage:

    @contextmanager
    def some_generator(<arguments>):
        <setup>
        try:
            yield <value>
        finally:
            <cleanup>

This makes this:

    with some_generator(<arguments>) as <variable>:
        <body>

equivalent to this:

    <setup>
    try:
        <variable> = <value>
        <body>
    finally:
        <cleanup>

---

#### phy.apps.exceptionHandler


**`phy.apps.exceptionHandler(exception_type, exception, traceback)`**



---

#### phy.apps.format_exception


**`phy.apps.format_exception(etype, value, tb, limit=None, chain=True)`**

Format a stack trace and the exception information.

The arguments have the same meaning as the corresponding arguments
to print_exception().  The return value is a list of strings, each
ending in a newline and some containing internal newlines.  When
these lines are concatenated and printed, exactly the same text is
printed as does print_exception().

---

### phy.apps.BaseController

Base controller for manual clustering GUI.

**Constructor**

* `dir_path : str or Path`
    Path to the data directory

* `config_dir : str or Path`
    Path to the configuration directory

* `model : Model`
    Model object, optional (it is automatically created otherwise)

* `plugins : list`
    List of plugins to manually activate, optional (the plugins are automatically loaded from
    the user configuration directory).

* `clear_cache : boolean`
    Whether to clear the cache on startup.

* `clear_state : boolean`
    Whether to clear the GUI state files on startup.

* `enable_threading : boolean`
    Whether to enable threading in the views when selecting clusters.

**Methods to override**

The main methods that can be overriden when implementing a custom `Controller` are:


* `_create_model() : None => object`
    Return a Model instance (any object, see below) from the controller constructor's
    parameters.

* `_set_view_creator() : None => None`
    Populate the `self.view_creator` dictionary with custom views.

* `get_best_channels(cluster_id) : int => list`
    Return the list of best channels for any given cluster, sorted by decreasing match.

**Model**

The Model can be any object, but it needs to implement the following properties and methods
in order to work with the BaseController:


* `channel_mapping : array-like`
    A `(n_channels,)` array with the column index in the raw data array of every channel.
    The displayed channel label of channel `channel_id` is `channel_mapping[channel_id]`.

* `channel_positions : array-like`
    A `(n_channels, 2)` array with the x, y coordinates of the electrode sites,
    in any unit (e.g. μm).

* `channel_probes : array-like (optional)`
    An `(n_channels,)` array with the probe index of every channel.

* `channel_shanks : array-like (optional)`
    An `(n_channels,)` array with the shank index of every channel (every probe might have
    multiple shanks). The shank index is relative to the probe. The pair (probe, shank)
    identifies uniquely a shank.

* `duration : float`
    The total duration of the recording, in seconds.

* `features : array-like`
    The object containing the features. The feature view is shown if this object is not None.

* `metadata : dict`
    Cluster metadata. Map metadata field names to dictionaries {cluster_id: value}.
    It is only expected to hold information representative of the state of the dataset
    on disk, not during a live clustering session.
    The special metadata field name `group` is reserved to cluster groups.

* `n_channels : int`
    Total number of channels in the recording (number of columns in the raw data array).

* `n_samples_waveforms : int`
    Number of time samples to use when extracting raw waveforms.

* `sample_rate : float`
    The sampling rate of the raw data.

* `spike_attributes : dict`
    Map attribute names to spike attributes, arrays of shape `(n_spikes,)`.

* `spike_clusters : array-like`
    Initial spike-cluster assignments, shape `(n_spikes,)`.

* `spike_samples : array-like`
    Spike samples, in samples, shape `(n_spikes,)`.

* `spike_times : array-like`
    Spike times, in seconds, shape `(n_spikes,)`.

* `spike_waveforms : Bunch`
    Extracted raw waveforms for a subset of the spikes.
    Should have attributes spike_ids, spike_channels, waveforms.

* `traces : array-like`
    Array (can be virtual/memmapped) of shape `(n_samples_total, n_channels)` with the
    raw data. The trace view is shown if this object is not None.

get_features(spike_ids, channel_ids) : array-like, array-like => array-like
    Return spike features of specified spikes on the specified channels. Optional.
get_waveforms(spike_ids, channel_ids) : array-like, array-like => array-like
    Return raw spike waveforms of specified spikes on the specified channels. Optional.


* `save_spike_clusters(spike_clusters) : array-like => None`
    Save spike clusters assignments back to disk.
save_metadata(name, values) : str, dict => None
    Save cluster metadata, where name is the metadata field name, and values a dictionary
    `{cluster_id: value}`.

**Note**

The Model represents data as it is stored on disk. When cluster data changes during
a manual clustering session (like spike-cluster assignments), the data in the model
is not expected to change (it is rather the responsability of the controller).

The model implements saving option for spike cluster assignments and cluster metadata.

---

#### BaseController.at_least_one_view


**`BaseController.at_least_one_view(self, view_name)`**

Add a view of a given type if there is not already one.

To be called before creating a GUI.

---

#### BaseController.create_amplitude_view


**`BaseController.create_amplitude_view(self)`**

Create the amplitude view.

---

#### BaseController.create_cluster_scatter_view


**`BaseController.create_cluster_scatter_view(self)`**

Create a cluster scatter view.

---

#### BaseController.create_correlogram_view


**`BaseController.create_correlogram_view(self)`**

Create a correlogram view.

---

#### BaseController.create_gui


**`BaseController.create_gui(self, default_views=None, **kwargs)`**

Create the GUI.

**Constructor**


* `default_views : list`
    List of views to add in the GUI, optional. By default, all views from the view
    count are added.

---

#### BaseController.create_ipython_view


**`BaseController.create_ipython_view(self)`**

Create an IPython View.

---

#### BaseController.create_misc_actions


**`BaseController.create_misc_actions(self, gui)`**



---

#### BaseController.create_probe_view


**`BaseController.create_probe_view(self)`**

Create a probe view.

---

#### BaseController.create_raster_view


**`BaseController.create_raster_view(self)`**

Create a raster view.

---

#### BaseController.get_background_spike_ids


**`BaseController.get_background_spike_ids(self, n=None)`**

Return regularly spaced spikes.

---

#### BaseController.get_best_channel


**`BaseController.get_best_channel(self, cluster_id)`**

Return the best channel id of a given cluster. This is the first channel returned
by `get_best_channels()`.

---

#### BaseController.get_best_channel_label


**`BaseController.get_best_channel_label(self, cluster_id)`**

Return the channel label of the best channel, for display in the cluster view.

---

#### BaseController.get_best_channels


**`BaseController.get_best_channels(self, cluster_id)`**

Return the best channels of a given cluster. To be overriden.

---

#### BaseController.get_channel_amplitudes


**`BaseController.get_channel_amplitudes(self, cluster_id)`**

Return the best channels of a given cluster along with their relative amplitudes.
To be overriden.

---

#### BaseController.get_channel_shank


**`BaseController.get_channel_shank(self, cluster_id)`**

Return the shank of a cluster's best channel, if the channel_shanks array is available.

---

#### BaseController.get_clusters_on_channel


**`BaseController.get_clusters_on_channel(self, channel_id)`**

Return all clusters which have the specified channel among their best channels.

---

#### BaseController.get_mean_firing_rate


**`BaseController.get_mean_firing_rate(self, cluster_id)`**

Return the mean firing rate of a cluster.

---

#### BaseController.get_probe_depth


**`BaseController.get_probe_depth(self, cluster_id)`**

Return the depth of a cluster.

---

#### BaseController.get_spike_ids


**`BaseController.get_spike_ids(self, cluster_id, n=None, **kwargs)`**

Return part or all of spike ids belonging to a given cluster.

---

#### BaseController.get_spike_times


**`BaseController.get_spike_times(self, cluster_id, n=None)`**

Return the spike times of spikes returned by `get_spike_ids(cluster_id, n)`.

---

#### BaseController.on_save_clustering


**`BaseController.on_save_clustering(self, sender, spike_clusters, groups, *labels)`**

Save the modified data.

---

#### BaseController.peak_channel_similarity


**`BaseController.peak_channel_similarity(self, cluster_id)`**

Return the list of similar clusters to a given cluster, just on the basis of the
peak channel.

**Parameters**

* `cluster_id : int`

**Returns**

* `similarities : list`
    List of tuples `(other_cluster_id, similarity_value)` sorted by decreasing
    similarity value.

---

### phy.apps.FeatureMixin



---

#### FeatureMixin.create_amplitude_view


**`FeatureMixin.create_amplitude_view(self)`**



---

#### FeatureMixin.create_feature_view


**`FeatureMixin.create_feature_view(self)`**



---

#### FeatureMixin.get_spike_feature_amplitudes


**`FeatureMixin.get_spike_feature_amplitudes(self, spike_ids, channel_id=None, channel_ids=None, pc=None, **kwargs)`**

Return the features for the specified channel and PC.

---

### phy.apps.Path

PurePath subclass that can make system calls.

Path represents a filesystem path but unlike PurePath, also offers
methods to do system calls on path objects. Depending on your system,
instantiating a Path will return either a PosixPath or a WindowsPath
object. You can also instantiate a PosixPath or WindowsPath directly,
but cannot instantiate a WindowsPath on a POSIX system or vice versa.

---

#### Path.None


**`Path.None`**

attrgetter(attr, ...) --> attrgetter object

Return a callable object that fetches the given attribute(s) from its operand.
After f = attrgetter('name'), the call f(r) returns r.name.
After g = attrgetter('name', 'date'), the call g(r) returns (r.name, r.date).
After h = attrgetter('name.first', 'name.last'), the call h(r) returns
(r.name.first, r.name.last).

---

#### Path.None


**`Path.None`**

attrgetter(attr, ...) --> attrgetter object

Return a callable object that fetches the given attribute(s) from its operand.
After f = attrgetter('name'), the call f(r) returns r.name.
After g = attrgetter('name', 'date'), the call g(r) returns (r.name, r.date).
After h = attrgetter('name.first', 'name.last'), the call h(r) returns
(r.name.first, r.name.last).

---

### phy.apps.QtDialogLogger

Display a message box for all errors.

---

#### QtDialogLogger.emit


**`QtDialogLogger.emit(self, record)`**

Do whatever it takes to actually log the specified logging record.

This version is intended to be implemented by subclasses and so
raises a NotImplementedError.

---

### phy.apps.TemplateMixin

Support templates.

The model needs to implement specific properties and methods.

amplitudes : array-like
    The template amplitude of every spike (only with TemplateMixin).
n_templates : int
    Initial number of templates.
spike_templates : array-like
    The template initial id of every spike.
get_template(template_id) : int => Bunch(template, channel_ids)
    Return the template data as a `(n_samples, n_channels)` array, the corresponding
    channel ids of the template.

---

#### TemplateMixin.create_template_view


**`TemplateMixin.create_template_view(self)`**

Create a template view.

---

#### TemplateMixin.get_amplitudes


**`TemplateMixin.get_amplitudes(self, cluster_id, load_all=False)`**

Return the spike amplitudes found in `amplitudes.npy`, for a given cluster.

---

#### TemplateMixin.get_cluster_amplitude


**`TemplateMixin.get_cluster_amplitude(self, cluster_id)`**

Return the amplitude of the best template of a cluster.

---

#### TemplateMixin.get_mean_spike_template_amplitudes


**`TemplateMixin.get_mean_spike_template_amplitudes(self, cluster_id)`**

Return the average of the spike template amplitudes.

---

#### TemplateMixin.get_spike_template_amplitudes


**`TemplateMixin.get_spike_template_amplitudes(self, spike_ids, **kwargs)`**

Return the spike template amplitudes as stored in `amplitudes.npy`.

---

#### TemplateMixin.get_spike_template_features


**`TemplateMixin.get_spike_template_features(self, spike_ids, first_cluster=None, **kwargs)`**

Return the template features of the requested spikes onto the first selected
cluster.

This is "the dot product (projection) of each spike waveform onto the template of the
first cluster."

See @mswallac's comment at
https://github.com/cortex-lab/phy/issues/868#issuecomment-520032905

---

#### TemplateMixin.get_template_amplitude


**`TemplateMixin.get_template_amplitude(self, template_id)`**

Return the maximum amplitude of a template's waveforms across all channels.

---

#### TemplateMixin.get_template_counts


**`TemplateMixin.get_template_counts(self, cluster_id)`**

Return a histogram of the number of spikes in each template for a given cluster.

---

#### TemplateMixin.get_template_for_cluster


**`TemplateMixin.get_template_for_cluster(self, cluster_id)`**

Return the largest template associated to a cluster.

---

### phy.apps.TraceMixin



---

#### TraceMixin.create_trace_image_view


**`TraceMixin.create_trace_image_view(self)`**

Create a trace image view.

---

#### TraceMixin.create_trace_view


**`TraceMixin.create_trace_view(self)`**

Create a trace view.

---

### phy.apps.WaveformMixin



---

#### WaveformMixin.create_waveform_view


**`WaveformMixin.create_waveform_view(self)`**



---

#### WaveformMixin.get_mean_spike_raw_amplitudes


**`WaveformMixin.get_mean_spike_raw_amplitudes(self, cluster_id)`**

Return the average of the spike raw amplitudes.

---

#### WaveformMixin.get_spike_raw_amplitudes


**`WaveformMixin.get_spike_raw_amplitudes(self, spike_ids, channel_id=None, **kwargs)`**

Return the maximum amplitude of the raw waveforms on the best channel of
the first selected cluster.

If `channel_id` is not specified, the returned amplitudes may be null.

---

## phy.apps.template

Template GUI.

---

#### phy.apps.template.from_sparse


**`phy.apps.template.from_sparse(data, cols, channel_ids)`**

Convert a sparse structure into a dense one.

**Parameters**


* `data : array-like`
    A (n_spikes, n_channels_loc, ...) array with the data.

* `cols : array-like`
    A (n_spikes, n_channels_loc) array with the channel indices of
    every row in data.

* `channel_ids : array-like`
    List of requested channel ids (columns).

---

#### phy.apps.template.get_template_params


**`phy.apps.template.get_template_params(params_path)`**

Get a dictionary of parameters from a `params.py` file.

---

#### phy.apps.template.load_model


**`phy.apps.template.load_model(params_path)`**

Return a TemplateModel instance from a path to a `params.py` file.

---

#### phy.apps.template.template_describe


**`phy.apps.template.template_describe(params_path)`**

Describe a template dataset.

---

#### phy.apps.template.template_gui


**`phy.apps.template.template_gui(params_path, **kwargs)`**

Launch the Template GUI.

---

### phy.apps.template.TemplateController

Controller for the Template GUI.

**Constructor**

* `dir_path : str or Path`
    Path to the data directory

* `config_dir : str or Path`
    Path to the configuration directory

* `model : Model`
    Model object, optional (it is automatically created otherwise)

* `plugins : list`
    List of plugins to manually activate, optional (the plugins are automatically loaded from
    the user configuration directory).

* `clear_cache : boolean`
    Whether to clear the cache on startup.

* `enable_threading : boolean`
    Whether to enable threading in the views when selecting clusters.

---

#### TemplateController.at_least_one_view


**`TemplateController.at_least_one_view(self, view_name)`**

Add a view of a given type if there is not already one.

To be called before creating a GUI.

---

#### TemplateController.create_amplitude_view


**`TemplateController.create_amplitude_view(self)`**



---

#### TemplateController.create_cluster_scatter_view


**`TemplateController.create_cluster_scatter_view(self)`**

Create a cluster scatter view.

---

#### TemplateController.create_correlogram_view


**`TemplateController.create_correlogram_view(self)`**

Create a correlogram view.

---

#### TemplateController.create_feature_view


**`TemplateController.create_feature_view(self)`**



---

#### TemplateController.create_gui


**`TemplateController.create_gui(self, default_views=None, **kwargs)`**

Create the GUI.

**Constructor**


* `default_views : list`
    List of views to add in the GUI, optional. By default, all views from the view
    count are added.

---

#### TemplateController.create_ipython_view


**`TemplateController.create_ipython_view(self)`**

Create an IPython View.

---

#### TemplateController.create_misc_actions


**`TemplateController.create_misc_actions(self, gui)`**



---

#### TemplateController.create_probe_view


**`TemplateController.create_probe_view(self)`**

Create a probe view.

---

#### TemplateController.create_raster_view


**`TemplateController.create_raster_view(self)`**

Create a raster view.

---

#### TemplateController.create_template_feature_view


**`TemplateController.create_template_feature_view(self)`**



---

#### TemplateController.create_template_view


**`TemplateController.create_template_view(self)`**

Create a template view.

---

#### TemplateController.create_trace_image_view


**`TemplateController.create_trace_image_view(self)`**

Create a trace image view.

---

#### TemplateController.create_trace_view


**`TemplateController.create_trace_view(self)`**

Create a trace view.

---

#### TemplateController.create_waveform_view


**`TemplateController.create_waveform_view(self)`**



---

#### TemplateController.get_amplitudes


**`TemplateController.get_amplitudes(self, cluster_id, load_all=False)`**

Return the spike amplitudes found in `amplitudes.npy`, for a given cluster.

---

#### TemplateController.get_background_spike_ids


**`TemplateController.get_background_spike_ids(self, n=None)`**

Return regularly spaced spikes.

---

#### TemplateController.get_best_channel


**`TemplateController.get_best_channel(self, cluster_id)`**

Return the best channel id of a given cluster. This is the first channel returned
by `get_best_channels()`.

---

#### TemplateController.get_best_channel_label


**`TemplateController.get_best_channel_label(self, cluster_id)`**

Return the channel label of the best channel, for display in the cluster view.

---

#### TemplateController.get_best_channels


**`TemplateController.get_best_channels(self, cluster_id)`**

Return the best channels of a given cluster.

---

#### TemplateController.get_channel_amplitudes


**`TemplateController.get_channel_amplitudes(self, cluster_id)`**

Return the channel amplitudes of the best channels of a given cluster.

---

#### TemplateController.get_channel_shank


**`TemplateController.get_channel_shank(self, cluster_id)`**

Return the shank of a cluster's best channel, if the channel_shanks array is available.

---

#### TemplateController.get_cluster_amplitude


**`TemplateController.get_cluster_amplitude(self, cluster_id)`**

Return the amplitude of the best template of a cluster.

---

#### TemplateController.get_clusters_on_channel


**`TemplateController.get_clusters_on_channel(self, channel_id)`**

Return all clusters which have the specified channel among their best channels.

---

#### TemplateController.get_mean_firing_rate


**`TemplateController.get_mean_firing_rate(self, cluster_id)`**

Return the mean firing rate of a cluster.

---

#### TemplateController.get_mean_spike_raw_amplitudes


**`TemplateController.get_mean_spike_raw_amplitudes(self, cluster_id)`**

Return the average of the spike raw amplitudes.

---

#### TemplateController.get_mean_spike_template_amplitudes


**`TemplateController.get_mean_spike_template_amplitudes(self, cluster_id)`**

Return the average of the spike template amplitudes.

---

#### TemplateController.get_probe_depth


**`TemplateController.get_probe_depth(self, cluster_id)`**

Return the depth of a cluster.

---

#### TemplateController.get_spike_feature_amplitudes


**`TemplateController.get_spike_feature_amplitudes(self, spike_ids, channel_id=None, channel_ids=None, pc=None, **kwargs)`**

Return the features for the specified channel and PC.

---

#### TemplateController.get_spike_ids


**`TemplateController.get_spike_ids(self, cluster_id, n=None, **kwargs)`**

Return part or all of spike ids belonging to a given cluster.

---

#### TemplateController.get_spike_raw_amplitudes


**`TemplateController.get_spike_raw_amplitudes(self, spike_ids, channel_id=None, **kwargs)`**

Return the maximum amplitude of the raw waveforms on the best channel of
the first selected cluster.

If `channel_id` is not specified, the returned amplitudes may be null.

---

#### TemplateController.get_spike_template_amplitudes


**`TemplateController.get_spike_template_amplitudes(self, spike_ids, **kwargs)`**

Return the spike template amplitudes as stored in `amplitudes.npy`.

---

#### TemplateController.get_spike_template_features


**`TemplateController.get_spike_template_features(self, spike_ids, first_cluster=None, **kwargs)`**

Return the template features of the requested spikes onto the first selected
cluster.

This is "the dot product (projection) of each spike waveform onto the template of the
first cluster."

See @mswallac's comment at
https://github.com/cortex-lab/phy/issues/868#issuecomment-520032905

---

#### TemplateController.get_spike_times


**`TemplateController.get_spike_times(self, cluster_id, n=None)`**

Return the spike times of spikes returned by `get_spike_ids(cluster_id, n)`.

---

#### TemplateController.get_template_amplitude


**`TemplateController.get_template_amplitude(self, template_id)`**

Return the maximum amplitude of a template's waveforms across all channels.

---

#### TemplateController.get_template_counts


**`TemplateController.get_template_counts(self, cluster_id)`**

Return a histogram of the number of spikes in each template for a given cluster.

---

#### TemplateController.get_template_for_cluster


**`TemplateController.get_template_for_cluster(self, cluster_id)`**

Return the largest template associated to a cluster.

---

#### TemplateController.on_save_clustering


**`TemplateController.on_save_clustering(self, sender, spike_clusters, groups, *labels)`**

Save the modified data.

---

#### TemplateController.peak_channel_similarity


**`TemplateController.peak_channel_similarity(self, cluster_id)`**

Return the list of similar clusters to a given cluster, just on the basis of the
peak channel.

**Parameters**

* `cluster_id : int`

**Returns**

* `similarities : list`
    List of tuples `(other_cluster_id, similarity_value)` sorted by decreasing
    similarity value.

---

#### TemplateController.template_similarity


**`TemplateController.template_similarity(self, cluster_id)`**

Return the list of similar clusters to a given cluster.

---

### phy.apps.template.TemplateModel

Object holding all data of a KiloSort/phy dataset.

**Constructor**


* `dir_path : str or Path`
    Path to the dataset directory

* `dat_path : str, Path, or list`
    Path to the raw data files.

* `dtype : NumPy dtype`
    Data type of the raw data file

* `offset : int`
    Header offset of the binary file

* `n_channels_dat : int`
    Number of channels in the dat file

* `sample_rate : float`
    Sampling rate of the data file.

---

#### TemplateModel.close


**`TemplateModel.close(self)`**

Close all memmapped files.

---

#### TemplateModel.describe


**`TemplateModel.describe(self)`**

Display basic information about the dataset.

---

#### TemplateModel.get_cluster_channels


**`TemplateModel.get_cluster_channels(self, cluster_id)`**

Return the most relevant channels of a cluster.

---

#### TemplateModel.get_cluster_mean_waveforms


**`TemplateModel.get_cluster_mean_waveforms(self, cluster_id)`**

Return the mean template waveforms of a cluster, as a weighted average of the
template waveforms from which the cluster originates from.

---

#### TemplateModel.get_cluster_spike_waveforms


**`TemplateModel.get_cluster_spike_waveforms(self, cluster_id)`**

Return all spike waveforms of a cluster, on the most relevant channels.

---

#### TemplateModel.get_cluster_spikes


**`TemplateModel.get_cluster_spikes(self, cluster_id)`**

Return the spike ids that belong to a given template.

---

#### TemplateModel.get_features


**`TemplateModel.get_features(self, spike_ids, channel_ids)`**

Return sparse features for given spikes.

---

#### TemplateModel.get_template


**`TemplateModel.get_template(self, template_id, channel_ids=None, amplitude_threshold=None)`**

Get data about a template.

---

#### TemplateModel.get_template_channels


**`TemplateModel.get_template_channels(self, template_id)`**

Return the most relevant channels of a template.

---

#### TemplateModel.get_template_counts


**`TemplateModel.get_template_counts(self, cluster_id)`**

Return a histogram of the number of spikes in each template for a given cluster.

---

#### TemplateModel.get_template_features


**`TemplateModel.get_template_features(self, spike_ids)`**

Return sparse template features for given spikes.

---

#### TemplateModel.get_template_spike_waveforms


**`TemplateModel.get_template_spike_waveforms(self, template_id)`**

Return all spike waveforms of a template, on the most relevant channels.

---

#### TemplateModel.get_template_spikes


**`TemplateModel.get_template_spikes(self, template_id)`**

Return the spike ids that belong to a given template.

---

#### TemplateModel.get_template_waveforms


**`TemplateModel.get_template_waveforms(self, template_id)`**

Return the waveforms of a template on the most relevant channels.

---

#### TemplateModel.get_waveforms


**`TemplateModel.get_waveforms(self, spike_ids, channel_ids=None)`**

Return spike waveforms on specified channels.

---

#### TemplateModel.save_metadata


**`TemplateModel.save_metadata(self, name, values)`**

Save a dictionary {cluster_id: value} with cluster metadata in
a TSV file.

---

#### TemplateModel.save_spike_clusters


**`TemplateModel.save_spike_clusters(self, spike_clusters)`**

Save the spike clusters.

---

#### TemplateModel.save_spikes_subset_waveforms


**`TemplateModel.save_spikes_subset_waveforms(self, max_n_spikes_per_template=None, max_n_channels=None)`**



---

#### TemplateModel.templates_amplitudes


**`TemplateModel.templates_amplitudes`**

Returns the average amplitude per cluster

---

#### TemplateModel.templates_channels


**`TemplateModel.templates_channels`**

Returns a vector of peak channels for all templates

---

#### TemplateModel.templates_probes


**`TemplateModel.templates_probes`**

Returns a vector of probe index for all templates

---

#### TemplateModel.templates_waveforms_durations


**`TemplateModel.templates_waveforms_durations`**

Returns a vector of waveform durations (ms) for all templates

---

## phy.apps.kwik

Kwik GUI.

---

#### phy.apps.kwik.kwik_describe


**`phy.apps.kwik.kwik_describe(path, channel_group=None, clustering=None)`**

Describe a template dataset.

---

#### phy.apps.kwik.kwik_gui


**`phy.apps.kwik.kwik_gui(path, channel_group=None, clustering=None, **kwargs)`**

Launch the Kwik GUI.

---

### phy.apps.kwik.KwikController

Controller for the Kwik GUI.

**Constructor**

* `kwik_path : str or Path`
    Path to the kwik file

* `channel_group : int`
    The default channel group to load

* `clustering : str`
    The default clustering to load

* `config_dir : str or Path`
    Path to the configuration directory

* `model : Model`
    Model object, optional (it is automatically created otherwise)

* `plugins : list`
    List of plugins to manually activate, optional (the plugins are automatically loaded from
    the user configuration directory).

* `clear_cache : boolean`
    Whether to clear the cache on startup.

* `enable_threading : boolean`
    Whether to enable threading in the views when selecting clusters.

---

#### KwikController.at_least_one_view


**`KwikController.at_least_one_view(self, view_name)`**

Add a view of a given type if there is not already one.

To be called before creating a GUI.

---

#### KwikController.create_amplitude_view


**`KwikController.create_amplitude_view(self)`**



---

#### KwikController.create_cluster_scatter_view


**`KwikController.create_cluster_scatter_view(self)`**

Create a cluster scatter view.

---

#### KwikController.create_correlogram_view


**`KwikController.create_correlogram_view(self)`**

Create a correlogram view.

---

#### KwikController.create_feature_view


**`KwikController.create_feature_view(self)`**



---

#### KwikController.create_gui


**`KwikController.create_gui(self, default_views=None, **kwargs)`**

Create the GUI.

**Constructor**


* `default_views : list`
    List of views to add in the GUI, optional. By default, all views from the view
    count are added.

---

#### KwikController.create_ipython_view


**`KwikController.create_ipython_view(self)`**

Create an IPython View.

---

#### KwikController.create_misc_actions


**`KwikController.create_misc_actions(self, gui)`**



---

#### KwikController.create_probe_view


**`KwikController.create_probe_view(self)`**

Create a probe view.

---

#### KwikController.create_raster_view


**`KwikController.create_raster_view(self)`**

Create a raster view.

---

#### KwikController.create_trace_image_view


**`KwikController.create_trace_image_view(self)`**

Create a trace image view.

---

#### KwikController.create_trace_view


**`KwikController.create_trace_view(self)`**

Create a trace view.

---

#### KwikController.create_waveform_view


**`KwikController.create_waveform_view(self)`**



---

#### KwikController.get_background_spike_ids


**`KwikController.get_background_spike_ids(self, n=None)`**

Return regularly spaced spikes.

---

#### KwikController.get_best_channel


**`KwikController.get_best_channel(self, cluster_id)`**

Return the best channel id of a given cluster. This is the first channel returned
by `get_best_channels()`.

---

#### KwikController.get_best_channel_label


**`KwikController.get_best_channel_label(self, cluster_id)`**

Return the channel label of the best channel, for display in the cluster view.

---

#### KwikController.get_best_channels


**`KwikController.get_best_channels(self, cluster_id)`**

Get the best channels of a given cluster.

---

#### KwikController.get_channel_amplitudes


**`KwikController.get_channel_amplitudes(self, cluster_id)`**

Return the channel amplitudes of the best channels of a given cluster.

---

#### KwikController.get_channel_shank


**`KwikController.get_channel_shank(self, cluster_id)`**

Return the shank of a cluster's best channel, if the channel_shanks array is available.

---

#### KwikController.get_clusters_on_channel


**`KwikController.get_clusters_on_channel(self, channel_id)`**

Return all clusters which have the specified channel among their best channels.

---

#### KwikController.get_mean_firing_rate


**`KwikController.get_mean_firing_rate(self, cluster_id)`**

Return the mean firing rate of a cluster.

---

#### KwikController.get_mean_spike_raw_amplitudes


**`KwikController.get_mean_spike_raw_amplitudes(self, cluster_id)`**

Return the average of the spike raw amplitudes.

---

#### KwikController.get_probe_depth


**`KwikController.get_probe_depth(self, cluster_id)`**

Return the depth of a cluster.

---

#### KwikController.get_spike_feature_amplitudes


**`KwikController.get_spike_feature_amplitudes(self, spike_ids, channel_id=None, channel_ids=None, pc=None, **kwargs)`**

Return the features for the specified channel and PC.

---

#### KwikController.get_spike_ids


**`KwikController.get_spike_ids(self, cluster_id, n=None, **kwargs)`**

Return part or all of spike ids belonging to a given cluster.

---

#### KwikController.get_spike_raw_amplitudes


**`KwikController.get_spike_raw_amplitudes(self, spike_ids, channel_id=None, **kwargs)`**

Return the maximum amplitude of the raw waveforms on the best channel of
the first selected cluster.

If `channel_id` is not specified, the returned amplitudes may be null.

---

#### KwikController.get_spike_times


**`KwikController.get_spike_times(self, cluster_id, n=None)`**

Return the spike times of spikes returned by `get_spike_ids(cluster_id, n)`.

---

#### KwikController.on_save_clustering


**`KwikController.on_save_clustering(self, sender, spike_clusters, groups, *labels)`**

Save the modified data.

---

#### KwikController.peak_channel_similarity


**`KwikController.peak_channel_similarity(self, cluster_id)`**

Return the list of similar clusters to a given cluster, just on the basis of the
peak channel.

**Parameters**

* `cluster_id : int`

**Returns**

* `similarities : list`
    List of tuples `(other_cluster_id, similarity_value)` sorted by decreasing
    similarity value.

---
