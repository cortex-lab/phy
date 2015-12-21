## Data utilities

The `phy.io` package contains utilities related to array manipulation, mock datasets, caching, and parallel computing context.

### Array

The `array` module contains functions to select subsets of large data arrays, and to obtain the spikes belonging to a set of clusters (notably the `Selector` class)

### Context

The `Context` provides facilities to accelerate computations through caching (with **joblib**) and parallel computing (with **ipyparallel**).

A `Context` is initialized with a cache directory (typically a subdirectory `.phy` within the directory containing the data). You can also provide an `ipy_view` instance to use parallel computing with the ipyparallel package.

#### Cache

Use `f = context.cache(f)` to cache a function. By default, the decorated function will be cached on disk in the cache directory, using joblib. NumPy arrays are fully and efficiently supported.

With the `memcache=True` argument, you can *also* use memory caching. This is interesting for example when caching functions returning a scalar for every cluster. This is the case with the functions computing the quality and similarity of clusters. These functions are called a lot during a manual clustering session.

#### Parallel computing

Use the `map()` and `map_async()` to call a function on multiple arguments in sequence or in parallel if an ipyparallel context is available (`ipy_view` keyword in the `Context`'s constructor).

There is also an experimental `map_dask_array(f, da)` method to map in parallel a function that processes a single chunk of a **dask Array**. The result of every computation unit is saved in a `.npy` file in the cache directory, and the result is a new dask Array that is dynamically memory-mapped from the stack of `.npy` files. **The cache directory needs to be available from all computing units for this method to work** (using a network file system). Doing it this way should mitigate the performance issues with transferring large amounts of data over the network.

#### Store

You can store JSON-serializable Python dictionaries with `context.save()` and `context.load()`. The files are saved in the cache directory. NumPy array and Qt buffers are fully supported. You can save the GUI state and geometry there for example.
