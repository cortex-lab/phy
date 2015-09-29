# -*- coding: utf-8 -*-

"""Execution context that handles parallel processing and cacheing."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op

import numpy as np

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _iter_chunks_dask(da):
    from dask.core import flatten
    for i, chunk in enumerate(flatten(da._keys())):
        yield i, chunk


def read_array(path):
    return np.load(path)


def write_array(path, arr):
    np.save(path, arr)


#------------------------------------------------------------------------------
# Context
#------------------------------------------------------------------------------

class Context(object):
    def __init__(self, cache_dir, ipy_client=None):
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
        self._ipy_client = ipy_client

    def _path(self, rel_path, *args, **kwargs):
        return op.join(self.cache_dir, rel_path.format(*args, **kwargs))

    def cache(self, f):
        if self._memory is None:  # pragma: no cover
            logger.debug("Joblib is not installed: skipping cacheing.")
            return
        return self._memory.cache(f)

    def map_dask_array(self, f, da, chunks=None, name=None,
                       dtype=None, shape=None):
        try:
            from dask.array import Array
            from dask.async import get_sync as get
        except ImportError:  # pragma: no cover
            raise Exception("dask is not installed. "
                            "Install it with `conda install dask`.")

        assert isinstance(da, Array)

        name = name or f.__name__
        assert name != da.name
        dtype = dtype or da.dtype
        shape = shape or da.shape
        chunks = chunks or da.chunks
        dask = da.dask

        def wrapped(chk):
            (i, chunk) = chk
            # Load the array's chunk.
            arr = get(dask, chunk)

            # Execute the function on the chunk.
            res = f(arr)

            # Save the output in the cache.
            if not op.exists(self._path(name)):
                os.makedirs(self._path(name))
            path = self._path('{name:s}/{i:d}.npy', name=name, i=i)
            write_array(path, res)

            # Return a dask pair to load the result.
            return (read_array, path)

        # Map the wrapped function normally.
        mapped = self.map(wrapped, _iter_chunks_dask(da))

        # Return the result as a dask array.
        dask = {(name, i): chunk for i, chunk in enumerate(mapped)}
        return Array(dask, name, chunks, dtype=dtype, shape=shape)

    def _map_serial(self, f, args):
        return [f(arg) for arg in args]

    def _map_ipy(self, f, args):
        return self._ipy_client.map(f, args)

    def map(self, f, args):
        if self._ipy_client:
            return self._map_ipy(f, args)
        else:
            return self._map_serial(f, args)
