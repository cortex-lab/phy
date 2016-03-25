# GUI

`phy.gui` provides generic Qt-based GUI components. You don't need to know Qt to use `phy.gui`, although it might help.

## Creating a Qt application

You need to create a Qt application before creating and using GUIs. There is a single Qt application object in a Python interpreter.

In IPython (console or notebook), you can just use the following magic command before doing anything Qt-related:

```python
>>> %gui qt
```

In other situations, like in regular Python scripts, you need to do two things:

* Call `phy.gui.create_app()` once, before you create a GUI.
* Call `phy.gui.run_app()` to launch your application. This blocks the Python interpreter and runs the Qt event loop. Generally, when this call returns, the application exits.

For interactive use and explorative work, it is highly recommended to use IPython, for example with a Jupyter Notebook.

## Creating a GUI

phy provides a **GUI**, a main window with dockable widgets (`QMainWindow`). By default, a GUI is empty, but you can add views. A view is any Qt widget, which includes HTML widgets, matplotlib figures, and VisPy canvases.

Let's create an empty GUI:

```python
>>> from phy.gui import GUI
>>> gui = GUI(position=(400, 200), size=(600, 400))
>>> gui.show()
```

## Adding a visualization

We can add any Qt widget with `gui.add_view(widget)`, as well as visualizations with VisPy or matplotlib (which are fully compatible with Qt and phy).

### With VisPy

The `gui.add_view()` method accepts any VisPy canvas. For example, here we add an empty VisPy window:

```python
>>> from vispy.app import Canvas
>>> from vispy import gloo
...
>>> c = Canvas()
...
>>> @c.connect
... def on_draw(e):
...     gloo.clear('purple')
...
>>> gui.add_view(c)
<phy.gui.gui.DockWidget at 0x7f68cefdfca8>
```

We can now dock and undock our widget from the GUI. This is particularly convenient when there are many widgets.

### With matplotlib

Here we add a matplotlib figure to our GUI:

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
...
>>> f = plt.figure()
>>> ax = f.add_subplot(111)
>>> t = np.linspace(-10., 10., 1000)
>>> ax.plot(t, np.sin(t))
>>> gui.add_view(f)
<phy.gui.gui.DockWidget at 0x7f68cd4c5ca8>
```

## Adding an HTML widget

phy provides an `HTMLWidget` component which allows you to create widgets in HTML. This is just a `QWebView` with some user-friendly facilities.

First, let's create a standalone HTML widget:

```python
>>> from phy.gui import HTMLWidget
>>> widget = HTMLWidget()
>>> widget.set_body("Hello world!")
>>> widget.show()
```

Now that our widget is created, let's add it to the GUI:

```python
>>> gui.add_view(widget)
<phy.gui.gui.DockWidget at 0x7f68cd4c5f78>
```

You'll find in the API reference other methods to edit the styles, scripts, header, and body of the HTML widget.

### Table

phy also provides a `Table` widget written in HTML and Javascript (using the [tablesort](https://github.com/tristen/tablesort) Javascript library). This widget shows a table of items, where every item (row) has an id, and every column is defined as a function `id => string`, the string being the contents of a row's cell in the table. The table can be sorted by every column.

One or several items can be selected by the user. The `select` event is raised when rows are selected. Here is a complete example:

```python
>>> from phy.gui import Table
>>> table = Table()
...
>>> # We add a column in the table.
... @table.add_column
... def name(id):
...     # This function takes an id as input and returns a string.
...     return "My id is %d" % id
...
>>> # Now we add some rows.
... table.set_rows([2, 3, 5, 7])
...
>>> # We print something when items are selected.
... @table.connect_
... def on_select(ids):
...     # NOTE: we use `connect_` and not `connect`, because `connect` is
...     # a Qt method associated to every Qt widget, and `Table` is a subclass
...     # of `QWidget`. Using `connect_` ensures that we're using phy's event
...     # system, not Qt's.
...     print("The items %s have been selected." % ids)
...
>>> table.show()
```

### Interactivity with Javascript

We can use Javascript in an HTML widget, and we can make Python and Javascript communicate.

```python
>>> from phy.gui import HTMLWidget
>>> widget = HTMLWidget()
>>> widget.set_body('<div id="mydiv"></div>')
>>> # We can execute Javascript code from Python.
... widget.eval_js("document.getElementById('mydiv').innerHTML='hello'")
>>> widget.show()
>>> gui.add_view(widget)

/home/cyrille/miniconda3/envs/phy/lib/python3.5/site-packages/matplotlib/axis.py:1015: UserWarning: Unable to find pixel distance along axis for interval padding of ticks; assuming no interval padding needed.
  warnings.warn("Unable to find pixel distance along axis "
/home/cyrille/miniconda3/envs/phy/lib/python3.5/site-packages/matplotlib/axis.py:1025: UserWarning: Unable to find pixel distance along axis for interval padding of ticks; assuming no interval padding needed.
  warnings.warn("Unable to find pixel distance along axis "<phy.gui.gui.DockWidget at 0x7f68cb315318>
```

You can use `widget.eval_js()` to evaluate Javascript code from Python. Conversely, you can use `widget.some_method()` from Javascript, where `some_method()` is a method implemented in your widget (which should be a subclass of `HTMLWidget`).

## Other GUI methods

Let's display the list of views in the GUI:

```python
>>> gui.list_views()
[<Canvas (PyQt4) at 0x7f68d3095f28>,
 <matplotlib.figure.Figure at 0x7f68cd4caf98>,
 <phy.gui.widgets.HTMLWidget at 0x7f68d3274ee8>,
 <phy.gui.widgets.HTMLWidget at 0x7f68cd505ca8>]
```

Use the following property to change the status bar:

```python
>>> gui.status_message = "Hello world"
```

Finally, the following methods allow you to save/restore the state of the GUI and the widgets:

```python
>>> gs = gui.save_geometry_state()
>>> gs
{'geometry': PyQt4.QtCore.QByteArray(b'\x01\xd9\xd0\xcb\x00\x01\x00\x00\x00\x00\x040\x00\x00\x00\xa2\x00\x00\x06\x9b\x00\x00\x03\x06\x00\x00\x04:\x00\x00\x00\xc8\x00\x00\x06\x91\x00\x00\x02\xfc\x00\x00\x00\x00\x00\x00'),
 'state': PyQt4.QtCore.QByteArray(b'\x00\x00\x00\xff\x00\x00\x00\x00\xfd\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x02X\x00\x00\x02\x06\xfc\x02\x00\x00\x00\x04\xfb\x00\x00\x00\x0e\x00C\x00a\x00n\x00v\x00a\x00s\x000\x01\x00\x00\x00\x19\x00\x00\x000\x00\x00\x00\x19\x00\xff\xff\xff\xfb\x00\x00\x00\x0e\x00F\x00i\x00g\x00u\x00r\x00e\x000\x01\x00\x00\x00O\x00\x00\x004\x00\x00\x00\x19\x00\xff\xff\xff\xfb\x00\x00\x00\x16\x00H\x00T\x00M\x00L\x00W\x00i\x00d\x00g\x00e\x00t\x000\x01\x00\x00\x00\x89\x00\x00\x00f\x00\x00\x00\x19\x00\xff\xff\xff\xfb\x00\x00\x00\x16\x00H\x00T\x00M\x00L\x00W\x00i\x00d\x00g\x00e\x00t\x001\x01\x00\x00\x00\xf5\x00\x00\x01*\x00\x00\x00\x19\x00\xff\xff\xff\x00\x00\x00\x00\x00\x00\x02\x06\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00\x08\xfc\x00\x00\x00\x00')}
```

```python
>>> gui.restore_geometry_state(gs)
```

The object `gs` is a JSON-serializable Python dictionary.

## Adding actions

An **action** is a Python function that the user can run from the menu bar or with a keyboard shortcut. You can create an `Actions` object to specify a list of actions attached to a GUI.

```python
>>> from phy.gui import Actions
>>> actions = Actions(gui)
...
>>> @actions.add(shortcut='ctrl+h')
... def hello():
...     print("Hello world!")
Hello world!
```

Now, if you press *Ctrl+H* in the GUI, you'll see `Hello world!` printed in the console.

Once an action is added, you can call it with `actions.hello()` where `hello` is the name of the action. By default, this is the name of the associated function, but you can also specify the name explicitly with the `name=...` keyword argument in `actions.add()`.

You'll find more details about `actions.add()` in the API reference. For example, use the `menu='MenuName'` keyword argument to add the action to a menu in the menu bar.

Every GUI comes with a `default_actions` property which implements actions always available in GUIs:

```python
>>> gui.default_actions
<Actions ['exit', 'show_all_shortcuts']>
```

For example, the following action shows the shortcuts of all actions attached to the GUI:

```python
>>> gui.default_actions.show_all_shortcuts()

Keyboard shortcuts for GUI - Default
- exit                                    : ctrl+q
- show_all_shortcuts                      : f1, h

Keyboard shortcuts for GUI - Snippets
- enable_snippet_mode                     : :

Keyboard shortcuts for GUI
- hello                                   : ctrl+h
```

You can create multiple `Actions` instance for a single GUI, which allows you to separate between different sets of actions.

## Snippets

The GUI provides a convenient system to quickly execute actions without leaving one's keyboard. Inspired by console-based text editors like *vim*, it is enabled by pressing `:` on the keyboard. Once this mode is enabled, what you type is displayed in the status bar. Then, you can call a function by typing its name or its alias. You can also use arguments to the actions, using a special syntax. Here is an example.

```python
>>> @actions.add(alias='c')
... def select(ids, obj):
...     print("Select %s with %s" % (ids, obj))
Select [3, 4, 5, 6] with hello
```

Now, pressing `:c 3-6 hello` followed by the `Enter` keystroke displays `Select [3, 4, 5, 6] with hello` in the console.

By convention, multiple arguments are separated by spaces, sequences of numbers are given either with `2,3,5,7` or `3-6` for consecutive numbers. If an alias is not specified when adding the action, you can always use the full action's name.

## GUI state

The **GUI state** is a special Python dictionary that holds info and parameters about a particular GUI session, like its position and size, the positions of the widgets, and other user preferences. This state is automatically persisted to disk (in JSON) in the config directory (passed as a parameter in the `create_gui()` function). By default, this is `~/.phy/gui_name/state.json`.

The GUI state is a `Bunch` instance, which derives from `dict` to support the additional `bunch.name` syntax.

Plugins can simply add fields to the GUI state and it will be persisted. There are special methods for GUI parameters: `state.save_gui_params()` and `state.load_gui_params()`.
