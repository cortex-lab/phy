# Plotting with VisPy

phy provides a simple and fast plotting system based on VisPy's low-level **gloo** interface. This plotting system is entirely generic. Currently, it privileges speed and scalability over quality. In other words, you can display millions of points at very high speed, but the plotting quality is not as good as matplotlib, for example. While this sytem uses the GPU extensively, knowledge of GPU or OpenGL is not required.

First, we need to activate the Qt event loop in IPython, or create and run the Qt application in a script.

```python
>>> %gui qt
```

## Simple view

Let's create a simple view with a scatter plot.

```python
>>> import numpy as np
>>> from phy.plot import View
```

```python
>>> view = View()
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

Several layouts are supported for subplots.

## Grid view

The Grid view lets you create multiple subplots arranged in a grid (like in matplotlib). Subplots are all individually clipped, which means that their viewports never overlap across the grid boundaries. Here is an example:

```python
>>> view = View(layout='grid', shape=(1, 2))  # the shape is `(n_rows, n_cols)`
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
>>> view = View(layout='stacked', n_plots=50)
...
>>> with view.building():
...     for i in range(view.n_plots):
...         view[i].plot(y=np.random.randn(2000),
...                      color=np.random.uniform(.5, .9, 4))
...
>>> view.show()
```

## Boxed view

The boxed view lets you put subplots at arbitrary locations. You can dynamically change the positions and the sizes of the boxes. An example is the waveform view, where line plots are positioned at the recording sites on a multielectrode array.

```python
>>> # Generate box positions along a circle.
... dt = np.pi / 10
>>> t = np.arange(0, 2 * np.pi, dt)
>>> x = np.cos(t)
>>> y = np.sin(t)
>>> box_pos = np.c_[x, y]
...
>>> view = View(layout='boxed', box_pos=box_pos)
...
>>> with view.building():
...     for i in range(view.n_plots):
...         # Create the subplots.
...         view[i].plot(y=np.random.randn(10, 100),
...                      color=np.random.uniform(.5, .9, 4))
...
>>> view.show()
```

## Data normalization

Data normalization is supported via the `data_bounds` keyword. This is a 4-tuple `(xmin, ymin, xmax, ymax)` with the coordinates of the viewport in the data coordinate system. By default, this is obtained with the min and max of the data. Here is an example:

```python
>>> view = View(layout='stacked', n_plots=2)
...
>>> n = 100
>>> x = np.linspace(0., 1., n)
>>> y = np.random.rand(n)
...
>>> with view.building():
...     view[0].plot(x, y, data_bounds=(0, 0, 1, 1))
...     view[1].plot(x, y, data_bounds=(0, -10, 1, 10))
...
>>> view.show()
```
