# GUI

`phy.gui` provides generic Qt-based GUI components. You don't need to know Qt to use `phy.gui`, although it might help.

## Creating a Qt application

You need to create a Qt application before creating and using GUIs. There is a single Qt application object in a Python interpreter.

In IPython (console or notebook), you can just use the following magic command before doing anything Qt-related:

```python
>>> %gui qt
```

In other situations, like in regular Python scripts, you need to:

* Call `phy.gui.create_app()` once, before you create a GUI.
* Call `phy.gui.run_app()` to launch your application. This blocks the Python interpreter and runs the Qt event loop. Generally, when this call returns, the application exits.

For interactive use and explorative work, it is highly recommended to use IPython, for example with a Jupyter Notebook.

## Creating a GUI

phy provides a **GUI**, a main window with dockable widgets (`QMainWindow`). By default, a GUI is empty, but you can add views. A view is any Qt widget or a VisPy canvas.

Let's create an empty GUI:

```python
>>> from phy.gui import GUI
>>> gui = GUI(position=(400, 200), size=(600, 400))
>>> gui.show()
INFO:phy.gui.actions:Snippet mode disabled.
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
<phy.gui.gui.DockWidget at 0x7f7e81466dc8>
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
...
>>> # TODO: implement this directly in phy
... from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
>>> gui.add_view(FigureCanvas(f))
<phy.gui.gui.DockWidget at 0x7f7e800da708>
```

## Adding an HTML widget

phy provides an `HTMLWidget` component which allows you to create widgets in HTML. This is just a `QWebView` with some user-friendly facilities.

First, let's create a standalone HTML widget:

```python
>>> from phy.gui.widgets import HTMLWidget
>>> widget = HTMLWidget()
>>> widget.set_body("Hello world!")
>>> widget.show()
```

Now that our widget is created, let's add it to the GUI:

```python
>>> gui.add_view(widget)
<phy.gui.gui.DockWidget at 0x7f7e780b3288>
```

You'll find in the API reference other methods to edit the styles, scripts, header, and body of the HTML widget.

### Interactivity with Javascript

We can use Javascript in an HTML widget, and we can make Python and Javascript communicate.

```python
>>> from phy.gui.widgets import HTMLWidget
>>> widget = HTMLWidget()
>>> widget.set_body('<div id="mydiv">')
>>> # We can execute Javascript code from Python.
... widget.eval_js("document.getElementById('mydiv').innerHTML='hello'")
>>> widget.show()
>>> gui.add_view(widget)
<phy.gui.gui.DockWidget at 0x7f7e780b3438>
```

You can use `widget.eval_js()` to evaluate Javascript code from Python. Conversely, you can use `widget.some_method()` from Javascript, where `some_method()` is a method implemented in a subclass of `HTMLWidget`.

## Other GUI methods

Let's display the list of views in the GUI:

```python
>>> gui.list_views()
[<phy.gui.gui.DockWidget at 0x7f7e81466dc8>,
 <phy.gui.gui.DockWidget at 0x7f7e800da708>,
 <phy.gui.gui.DockWidget at 0x7f7e780b3288>,
 <phy.gui.gui.DockWidget at 0x7f7e780b3438>]
```

The following method allows you to check how many views of each class there are:

```python
>>> gui.view_count()
{'canvas': 1, 'figurecanvasqtagg': 1, 'htmlwidget': 2}
```

Use the following property to change the status bar:

```python
>>> gui.status_message = "Hello world"
```

Finally, the following methods allow you to save/restore the state of the GUI and the widgets:

```python
>>> gs = gui.save_geometry_state()
```

```python
>>> gui.restore_geometry_state(gs)
```

The object `gs` is a JSON-serializable Python dictionary.

## Adding actions

An **action** is a Python function that can be run by the user by clicking on a button or pressing a keyboard shortcut. You can create an `Actions` object to specify a list of actions attached to a GUI.

```python
>>> from phy.gui import Actions
>>> actions = Actions(gui)
...
>>> @actions.add(shortcut='ctrl+h')
... def hello():
...     print("Hello world!")
```

Now, if you press *Ctrl+H* in the GUI, you'll see ̀`Hello world!` printed in the console.

Once an action is added, you can call it with `actions.hello()` where `hello` is the name of the action. By default, this is the name of the associated function, but you can also specify the name explicitly with the `name=...` keyword argument in `actions.add()`.

You'll find more details about `actions.add()` in the API reference.

Every GUI comes with a `default_actions` property which implements actions always available in GUIs:

```python
>>> gui.default_actions
<Actions ['exit', 'show_shortcuts']>
```

For example, the following action shows the shortcuts of all actions attached to the GUI:

```python
>>> gui.default_actions.show_shortcuts()

Keyboard shortcuts for GUI
enable_snippet_mode                     : :
exit                                    : ctrl+q
hello                                   : ctrl+h
show_shortcuts                          : f1, h
```

## Snippets

The GUI provides a convenient system to quickly execute actions without leaving one's keyboard. Inspired by console-based text editors like *vim*, it is enabled by pressing `:` on the keyboard. Once this mode is enabled, what you type is displayed in the status bar. Then, you can call a function by typing its name or its alias. You can also use arguments to the actions, using a special syntax. Here is an example.

```python
>>> @actions.add(alias='c')
... def select(ids, obj):
...     print("Select %s with %s" % (ids, obj))
```

Now, pressing `:c 3-6 hello` followed by the `Enter` keystrokes displays `Select [3, 4, 5, 6] with hello` in the console.

By convention, multiple arguments are separated by spaces, sequences of numbers are given either with `2,3,5,7` or `3-6` for consecutive numbers. If an alias is not specified when adding the action, you can always use the full action's name.
