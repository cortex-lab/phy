# Plotting with VisPy

phy provides a simple and fast plotting system based on VisPy's low-level **gloo** interface. This plotting system is entirely generic. Currently, it privileges speed and scalability over quality. In other words, you can display millions of points at very high speed, but the plotting quality is not as good as matplotlib, for example.

First, we need to activate the Qt event loop in IPython, or create and run the Qt application in a script.

```python
>>> %gui qt
```

## Simple view

Let's create a simple view with a scatter plot.

```python
>>> import numpy as np
>>> from phy.plot import SimpleView, GridView, BoxedView, StackedView
```

```python
>>> view = SimpleView()
...
>>> n = 1000
>>> x, y = np.random.randn(2, n)
>>> c = np.random.uniform(.3, .7, (n, 4))
>>> s = np.random.uniform(5, 30, n)
...
>>> # NOTE: currently, the building process needs to be explicit.
... # All commands that construct the view should be enclosed in this
... # context manager, or at least one should ensure that
... # `view.clear()` and `view.build()` are called before and after
... # the building commands.
... with view.building():
...     view.scatter(x, y, color=c, size=s, marker='disc')
...
>>> view.show()
```

Note that you can pan and zoom with the mouse and keyboard.

The other plotting commands currently supported are `plot()` and `hist()`. We're planning to add support for text in the near future.

## Grid view

The `GridView` lets you create multiple subplots arranged in a grid. Subplots are all individually clipped, which means that their viewports never overlap across the grid boundaries. Here is an example:

```python
>>> view = GridView((1, 2))  # the shape is `(n_rows, n_cols)`
...
>>> x = np.linspace(-10., 10., 1000)
...
>>> with view.building():
...     view[0, 0].plot(x, np.sin(x))
...     view[0, 1].plot(x, np.cos(x), color=(1, 0, 0, 1))
...
>>> view.show()
```

Subplots are created with the `view[i, j]` syntax. The indexing scheme works like mathematical matrices (origin at the upper left).

Note that there are no axes at this point, but we'll be working on it. Also, independent per-subplot panning and zooming is not supported and this is unlikely to change in the foreseable future.

## Stacked view

The stacked view lets you stack several subplots vertically with no clipping. An example is a trace view showing a multichannel time-dependent signal.

```python
>>> view = StackedView(50)
...
>>> with view.building():
...     for i in range(view.n_plots):
...         view[i].plot(y=np.random.randn(2000),
...                      color=np.random.uniform(.5, .9, 4))
...
>>> view.show()
```
