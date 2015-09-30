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
try:
    from dask.async import get_sync as get
except ImportError:  # pragma: no cover
    raise Exception("dask is not installed. "
                    "Install it with `conda install dask`.")

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _iter_chunks_dask(da):
    from dask.core import flatten
    for chunk in flatten(da._keys()):
        yield chunk


def read_array(path):
    return np.load(path)


def write_array(path, arr):
    np.save(path, arr)


#------------------------------------------------------------------------------
# Context
#------------------------------------------------------------------------------

def _mapped(i, chunk, dask, func, dirpath):
    # Load the array's chunk.
    arr = get(dask, chunk)

    # Execute the function on the chunk.
    res = func(arr)

    # Save the output in the cache.
    if not op.exists(dirpath):
        os.makedirs(dirpath)
    path = op.join(dirpath, '{}.npy'.format(i))
    write_array(path, res)

    # Return a dask pair to load the result.
    return (read_array, path)


class Context(object):
    def __init__(self, cache_dir, ipy_view=None):
        self.cache_dir = op.realpath(cache_dir)
        if not op.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        try:
            from joblib import Memory
            joblib_cachedir = self._path('joblib')
            self._memory = Memory(cachedir=joblib_cachedir, verbose=0)
        except ImportError:  # pragma: no cover
            logger.warn("Joblib is not installed. "
                        "Install it with `conda install joblib`.")
            self._memory = None
        self.ipy_view = ipy_view if ipy_view else None

    @property
    def ipy_view(self):
        return self._ipy_view

    @ipy_view.setter
    def ipy_view(self, value):
        self._ipy_view = value
        if hasattr(value, 'use_dill'):
            # Dill is necessary because we need to serialize closures.
            value.use_dill()

    def _path(self, rel_path, *args, **kwargs):
        return op.join(self.cache_dir, rel_path.format(*args, **kwargs))

    def cache(self, f):
        if self._memory is None:  # pragma: no cover
            logger.debug("Joblib is not installed: skipping cacheing.")
            return
        return self._memory.cache(f)

    def map_dask_array(self, func, da, chunks=None, name=None,
                       dtype=None, shape=None):
        try:
            from dask.array import Array
        except ImportError:  # pragma: no cover
            raise Exception("dask is not installed. "
                            "Install it with `conda install dask`.")

        assert isinstance(da, Array)

        name = name or func.__name__
        assert name != da.name
        dtype = dtype or da.dtype
        shape = shape or da.shape
        chunks = chunks or da.chunks
        dask = da.dask

        args_0 = list(_iter_chunks_dask(da))
        n = len(args_0)
        dirpath = op.join(self.cache_dir, name)
        mapped = self.map(_mapped, range(n), args_0, [dask] * n,
                          [func] * n, [dirpath] * n)

        with open(op.join(dirpath, 'info'), 'wb') as f:
            dump({'chunks': chunks, 'dtype': dtype, 'axis': 0}, f)

        # Return the result as a dask array.
        dask = {(name, i): chunk for i, chunk in enumerate(mapped)}
        return Array(dask, name, chunks, dtype=dtype, shape=shape)

    def _map_serial(self, f, *args):
        return [f(*arg) for arg in zip(*args)]

    def _map_ipy(self, f, *args, **kwargs):
        if kwargs.get('sync', True):
            name = 'map_sync'
        else:
            name = 'map_async'
        return getattr(self._ipy_view, name)(f, *args)

    def map_async(self, f, *args):
        if self._ipy_view:
            return self._map_ipy(f, *args, sync=False)
        else:
            return self._map_serial(f, *args)

    def map(self, f, *args):
        if self._ipy_view:
            return self._map_ipy(f, *args, sync=True)
        else:
            return self._map_serial(f, *args)
