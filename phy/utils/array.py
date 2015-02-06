# -*- coding: utf-8 -*-

"""Utility functions for NumPy arrays."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from math import floor

import numpy as np

from ..ext import six


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _range_from_slice(myslice, start=None, stop=None, step=None, length=None):
    """Convert a slice to an array of integers."""
    assert isinstance(myslice, slice)
    # Find 'step'.
    step = step if step is not None else myslice.step
    if step is None:
        step = 1
    # Find 'start'.
    start = start if start is not None else myslice.start
    if start is None:
        start = 0
    # Find 'stop' as a function of length if 'stop' is unspecified.
    stop = stop if stop is not None else myslice.stop
    if length is not None:
        stop_inferred = floor(start + step * length)
        if stop is not None and stop < stop_inferred:
            raise ValueError("'stop' ({stop}) and ".format(stop=stop) +
                             "'length' ({length}) ".format(length=length) +
                             "are not compatible.")
        stop = stop_inferred
    if stop is None and length is None:
        raise ValueError("'stop' and 'length' cannot be both unspecified.")
    myrange = np.arange(start, stop, step)
    # Check the length if it was specified.
    if length is not None:
        assert len(myrange) == length
    return myrange


def _unique(x):
    """Faster version of np.unique().
    This version is restricted to 1D arrays of non-negative integers.
    It is only faster if len(x) >> len(unique(x)).
    """
    if len(x) == 0:
        return np.array([], dtype=np.int)
    return np.nonzero(np.bincount(x))[0]


def _normalize(arr):
    """Normalize an array into [0, 1]."""
    # TODO: add 'keep_ratio' option.
    min, max = arr.min(axis=0), arr.max(axis=0)
    positions_n = (arr - min) * 1. / (max - min)
    return positions_n


def _index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.

    Implicitely assume that:

    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.

    This is not checked for performance reasons.

    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    # TODO: assertions to disable in production for performance reasons.
    m = lookup.max() + 1
    tmp = np.zeros(m, dtype=np.int)
    tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


_ACCEPTED_ARRAY_DTYPES = (np.float, np.float32, np.float64,
                          np.int, np.int8, np.int16, np.uint8, np.uint16,
                          np.int32, np.int64, np.uint32, np.uint64,
                          np.bool)


def _as_array(arr):
    """Convert an object to a numerical NumPy array.

    Avoid a copy if possible.

    """
    if isinstance(arr, six.integer_types + (float,)):
        arr = [arr]
    out = np.asarray(arr)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError("'arr' seems to have an invalid dtype: "
                         "{0:s}".format(str(out.dtype)))
    return out


# -----------------------------------------------------------------------------
# Chunking functions
# -----------------------------------------------------------------------------

def chunk_bounds(n_samples, chunk_size, overlap=0):
    """Return chunk bounds.

    Chunks have the form:

        [ overlap/2 | chunk_size-overlap | overlap/2 ]
        s_start   keep_start           keep_end     s_end

    Except for the first and last chunks which do not have a left/right
    overlap.

    This generator yields (s_start, s_end, keep_start, keep_end).

    """
    s_start = 0
    s_end = chunk_size
    keep_start = s_start
    keep_end = s_end - overlap // 2
    yield s_start, s_end, keep_start, keep_end

    while s_end - overlap + chunk_size < n_samples:
        s_start = s_end - overlap
        s_end = s_start + chunk_size
        keep_start = keep_end
        keep_end = s_end - overlap // 2
        if s_start < s_end:
            yield s_start, s_end, keep_start, keep_end

    s_start = s_end - overlap
    s_end = n_samples
    keep_start = keep_end
    keep_end = s_end
    if s_start < s_end:
        yield s_start, s_end, keep_start, keep_end


def _excerpt_step(n_samples, n_excerpts=None, excerpt_size=None):
    """Compute the step of an excerpt set as a function of the number
    of excerpts or their sizes."""
    step = max((n_samples - excerpt_size) // (n_excerpts - 1),
               excerpt_size)
    return step


def excerpts(n_samples, n_excerpts=None, excerpt_size=None):
    """Yield (start, end) where start is included and end is excluded."""
    step = _excerpt_step(n_samples,
                         n_excerpts=n_excerpts,
                         excerpt_size=excerpt_size)
    for i in range(n_excerpts):
        start = i * step
        if start >= n_samples:
            break
        end = min(start + excerpt_size, n_samples)
        yield start, end


def data_chunk(data, chunk, with_overlap=False):
    """Get a data chunk."""
    assert isinstance(chunk, tuple)
    if len(chunk) == 2:
        i, j = chunk
    elif len(chunk) == 4:
        if with_overlap:
            i, j = chunk[:2]
        else:
            i, j = chunk[2:]
    else:
        raise ValueError("'chunk' should have 2 or 4 elements, "
                         "not {0:d}".format(len(chunk)))
    return data[i:j, ...]


# -----------------------------------------------------------------------------
# PartialArray
# -----------------------------------------------------------------------------

def _as_tuple(item):
    """Ensure an item is a tuple."""
    if item is None:
        return None
    # elif hasattr(item, '__len__'):
    #     return tuple(item)
    elif not isinstance(item, tuple):
        return (item,)
    else:
        return item


def _partial_shape(shape, trailing_index):
    """Return the shape of a partial array."""
    if shape is None:
        shape = ()
    if trailing_index is None:
        trailing_index = ()
    trailing_index = _as_tuple(trailing_index)
    # Length of the selection items for the partial array.
    len_item = len(shape) - len(trailing_index)
    # Array for the trailing dimensions.
    _arr = np.empty(shape=shape[len_item:])
    try:
        trailing_arr = _arr[trailing_index]
    except IndexError:
        raise ValueError("The partial shape index is invalid.")
    return shape[:len_item] + trailing_arr.shape


class PartialArray(object):
    """Proxy to a view of an array, allowing selection along the first
    dimensions and fixing the trailing dimensions."""
    def __init__(self, arr, trailing_index=None):
        self._arr = arr
        self._trailing_index = _as_tuple(trailing_index)
        self.shape = _partial_shape(arr.shape, self._trailing_index)
        self.dtype = arr.dtype

    def __getitem__(self, item):
        if self._trailing_index is None:
            return self._arr[item]
        else:
            item = _as_tuple(item)
            item += self._trailing_index
            if len(item) != len(self._arr.shape):
                raise ValueError("The array selection is invalid: "
                                 "{0}".format(str(item)))
            return self._arr[item]
