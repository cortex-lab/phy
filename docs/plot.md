# Plotting with VisPy

phy provides a simple and fast plotting system based on VisPy's low-level **gloo** interface. This plotting system is entirely generic.

```python
>>> %gui qt
```

```python
>>> import numpy as np
>>> from phy.plot import SimpleView
```

```python
>>> view = SimpleView()
...
>>> n = 1000
>>> x, y = np.random.randn(2, n)
>>> c = np.random.uniform(.3, .7, (n, 4))
>>> s = np.random.uniform(5, 30, n)
...
>>> with view.building():
...     view.scatter(x, y, color=c, size=s, marker='disc')
...
>>> view.show()
```

```python

```
