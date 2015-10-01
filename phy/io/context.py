# -*- coding: utf-8 -*-

"""Execution context that handles parallel processing and cacheing."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op

import numpy as np
from six.moves.cPickle import dump
from six import string_types
try:
    from dask.array import Array
    from dask.async import get_sync as get
    from dask.core import flatten
except ImportError:  # pragma: no cover
    raise Exception("dask is not installed. "
                    "Install it with `conda install dask`.")

from .array import read_array, write_array
from phy.utils import Bunch

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _iter_chunks_dask(da):
    for chunk in flatten(da._keys()):
        yield chunk


#------------------------------------------------------------------------------
# Context
#------------------------------------------------------------------------------

def _mapped(i, chunk, dask, func, args, cache_dir, name):
    """Top-level function to map.

    This function needs to be a top-level function for ipyparallel to work.

    """
    # Load the array's chunk.
    arr = get(dask, chunk)

    # Execute the function on the chunk.
    # logger.debug("Run %s on chunk %d", name, i)
    res = func(arr, *args)

    # Save the result, and return the information about what we saved.
    return _save_stack_chunk(i, res, cache_dir, name)


def _save_stack_chunk(i, arr, cache_dir, name):
    """Save an output chunk array to a npy file, and return information about
    it."""
    # Handle the case where several output arrays are returned.
    if isinstance(arr, tuple):
        # The name is a tuple of names for the different arrays returned.
        assert isinstance(name, tuple)
        assert len(arr) == len(name)

        return tuple(_save_stack_chunk(i, arr_, cache_dir, name_)
                     for arr_, name_ in zip(arr, name))

    assert isinstance(name, string_types)
    assert isinstance(arr, np.ndarray)

    dirpath = op.join(cache_dir, name)
    path = op.join(dirpath, '{}.npy'.format(i))
    write_array(path, arr)

    # Return information about what we just saved.
    return Bunch(dask_tuple=(read_array, path),
                 shape=arr.shape,
                 dtype=arr.dtype,
                 name=name,
                 dirpath=dirpath,
                 )


def _save_stack_info(outputs):
    """Save the npy stack info, and return one or several dask arrays from
    saved npy stacks.

    The argument is a list of objects returned by `_save_stack_chunk()`.

    """
    # Handle the case where several arrays are returned, i.e. outputs is a list
    # of tuples of Bunch objects.
    assert len(outputs)
    if isinstance(outputs[0], tuple):
        return tuple(_save_stack_info(output) for output in zip(*outputs))

    # Get metadata fields common to all chunks.
    assert len(outputs)
    assert isinstance(outputs[0], Bunch)
    name = outputs[0].name
    dirpath = outputs[0].dirpath
    dtype = outputs[0].dtype
    trail_shape = outputs[0].shape[1:]
    trail_ndim = len(trail_shape)

    # Ensure the consistency of all chunks metadata.
    assert all(output.name == name for output in outputs)
    assert all(output.dirpath == dirpath for output in outputs)
    assert all(output.dtype == dtype for output in outputs)
    assert all(output.shape[1:] == trail_shape for output in outputs)

    # Compute the output dask array chunks and shape.
    chunks = (tuple(output.shape[0] for output in outputs),) + trail_shape
    n = sum(output.shape[0] for output in outputs)
    shape = (n,) + trail_shape

    # Save the info object for dask npy stack.
    with open(op.join(dirpath, 'info'), 'wb') as f:
        dump({'chunks': chunks, 'dtype': dtype, 'axis': 0}, f)

    # Return the result as a dask array.
    dask_tuples = tuple(output.dask_tuple for output in outputs)
    dask = {((name, i) + (0,) * trail_ndim): chunk
            for i, chunk in enumerate(dask_tuples)}
    return Array(dask, name, chunks, dtype=dtype, shape=shape)


def _ensure_cache_dirs_exist(cache_dir, name):
    if isinstance(name, tuple):
        return [_ensure_cache_dirs_exist(cache_dir, name_) for name_ in name]
    dirpath = op.join(cache_dir, name)
    if not op.exists(dirpath):
        os.makedirs(dirpath)


class Context(object):
    """Handle function cacheing and parallel map with ipyparallel."""
    def __init__(self, cache_dir, ipy_view=None):

        # Make sure the cache directory exists.
        self.cache_dir = op.realpath(cache_dir)
        if not op.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self._set_memory(cache_dir)
        self.ipy_view = ipy_view if ipy_view else None

    def _set_memory(self, cache_dir):
        # Try importing joblib.
        try:
            from joblib import Memory
            joblib_cachedir = self._path('joblib')
            self._memory = Memory(cachedir=joblib_cachedir)
        except ImportError:  # pragma: no cover
            logger.warn("Joblib is not installed. "
                        "Install it with `conda install joblib`.")
            self._memory = None

    @property
    def ipy_view(self):
        """ipyparallel view to parallel computing resources."""
        return self._ipy_view

    @ipy_view.setter
    def ipy_view(self, value):
        self._ipy_view = value
        if hasattr(value, 'use_dill'):
            # Dill is necessary because we need to serialize closures.
            value.use_dill()

    def _path(self, rel_path, *args, **kwargs):
        """Get the full path to a local cache resource."""
        return op.join(self.cache_dir, rel_path.format(*args, **kwargs))

    def cache(self, f):
        """Cache a function using the context's cache directory."""
        if self._memory is None:  # pragma: no cover
            logger.debug("Joblib is not installed: skipping cacheing.")
            return
        return self._memory.cache(f)

    def map_dask_array(self, func, da, *args, **kwargs):
        """Map a function on the chunks of a dask array, and return a
        new dask array.

        This function works in parallel if an `ipy_view` has been set.

        Every task loads one chunk, applies the function, and saves the
        result into a `<i>.npy` file in a cache subdirectory with the specified
        name (the function's name by default). The result is a new dask array
        that reads data from the npy stack in the cache subdirectory.

        The mapped function can return several arrays as a tuple. In this case,
        `name` must also be a tuple, and the output of this function is a
        tuple of dask arrays.

        """
        assert isinstance(da, Array)

        name = kwargs.get('name', None) or func.__name__
        assert name != da.name
        dask = da.dask

        # Ensure the directories exist.
        _ensure_cache_dirs_exist(self.cache_dir, name)

        args_0 = list(_iter_chunks_dask(da))
        n = len(args_0)
        output = self.map(_mapped, range(n), args_0, [dask] * n,
                          [func] * n, [args] * n,
                          [self.cache_dir] * n, [name] * n)

        # output contains information about the output arrays. We use this
        # information to reconstruct the final dask array.
        return _save_stack_info(output)

    def _map_serial(self, f, *args):
        return [f(*arg) for arg in zip(*args)]

    def _map_ipy(self, f, *args, **kwargs):
        if kwargs.get('sync', True):
            name = 'map_sync'
        else:
            name = 'map_async'
        return getattr(self._ipy_view, name)(f, *args)

    def map_async(self, f, *args):
        """Map a function asynchronously.

        Use the ipyparallel resources if available.

        """
        if self._ipy_view:
            return self._map_ipy(f, *args, sync=False)
        else:
            raise RuntimeError("Asynchronous execution requires an "
                               "ipyparallel context.")

    def map(self, f, *args):
        """Map a function synchronously.

        Use the ipyparallel resources if available.

        """
        if self._ipy_view:
            return self._map_ipy(f, *args, sync=True)
        else:
            return self._map_serial(f, *args)

    def __getstate__(self):
        """Make sure that this class is picklable."""
        state = self.__dict__.copy()
        state['_memory'] = None
        state['_ipy_view'] = None
        return state

    def __setstate__(self, state):
        """Make sure that this class is picklable."""
        self.__dict__ = state
        # Recreate the joblib Memory instance.
        self._set_memory(state['cache_dir'])