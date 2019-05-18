# -*- coding: utf-8 -*-

"""Plotting utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path

import numpy as np

from phylib.utils import Bunch

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Data validation
#------------------------------------------------------------------------------

def _get_texture(arr, default, n_items, from_bounds):
    """Prepare data to be uploaded as a texture.

    The from_bounds must be specified.

    """
    if not hasattr(default, '__len__'):  # pragma: no cover
        default = [default]
    n_cols = len(default)
    if arr is None:  # pragma: no cover
        arr = np.tile(default, (n_items, 1))
    assert arr.shape == (n_items, n_cols)
    # Convert to 3D texture.
    arr = arr[np.newaxis, ...].astype(np.float64)
    assert arr.shape == (1, n_items, n_cols)
    # NOTE: we need to cast the texture to [0., 1.] (float texture).
    # This is easy as soon as we assume that the signal bounds are in
    # [-1, 1].
    assert len(from_bounds) == 2
    m, M = map(float, from_bounds)
    assert np.all(arr >= m)
    assert np.all(arr <= M)
    arr = (arr - m) / (M - m)
    assert np.all(arr >= 0)
    assert np.all(arr <= 1.)
    return arr


def _get_array(val, shape, default=None, dtype=np.float64):
    """Ensure an object is an array with the specified shape."""
    assert val is not None or default is not None
    if hasattr(val, '__len__') and len(val) == 0:  # pragma: no cover
        val = None
    # Do nothing if the array is already correct.
    if (isinstance(val, np.ndarray) and
            val.shape == shape and
            val.dtype == dtype):
        return val
    out = np.zeros(shape, dtype=dtype)
    # This solves `ValueError: could not broadcast input array from shape (n)
    # into shape (n, 1)`.
    if val is not None and isinstance(val, np.ndarray):
        if val.size == out.size:
            val = val.reshape(out.shape)
    out.flat[:] = val if val is not None else default
    assert out.shape == shape
    return out


def _get_pos(x, y):
    assert x is not None
    assert y is not None

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Validate the position.
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape

    return x, y


def _get_index(n_items, item_size, n):
    """Prepare an index attribute for GPU uploading."""
    index = np.arange(n_items)
    index = np.repeat(index, item_size)
    index = index.astype(np.float64)
    assert index.shape == (n,)
    return index


def _get_linear_x(n_signals, n_samples):
    return np.tile(np.linspace(-1., 1., n_samples), (n_signals, 1))


class BatchAccumulator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.items = {}
        self.noconcat = ()

    def add(self, b, noconcat=()):
        self.noconcat = noconcat
        n = None
        # We assume that the first element of the b dictionary is an
        # ndarray, that sets the number of vertices for all elements
        # in the dictionary.
        for key, val in b.items():
            if key not in self.items:
                self.items[key] = []
            if val is None:
                continue
            # Size of the second dimension.
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    val = np.c_[val]
                assert val.ndim == 2
                n, k = val.shape
            elif isinstance(val, (tuple, list)):
                k = len(val)
            else:
                k = 1
            # Special consideration for list of strings (text visual).
            if key in noconcat:
                self.items[key].extend(val)
            else:
                assert n is not None  # n should had been set above, or in a previous iteration.
                self.items[key].append(_get_array(val, (n, k)))
        return b

    def __getattr__(self, key):
        if key not in self.items:
            raise AttributeError()
        arrs = self.items.get(key)
        if not arrs:
            return None
        # Special consideration for list of strings (text visual).
        if key in self.noconcat:
            return arrs
        return np.concatenate(arrs, axis=0)

    @property
    def data(self):
        return Bunch({key: getattr(self, key) for key in self.items.keys()})


#------------------------------------------------------------------------------
# Misc
#------------------------------------------------------------------------------

def _load_shader(filename):
    """Load a shader file."""
    return (Path(__file__).parent / 'glsl' / filename).read_text()


def _tesselate_histogram(hist):
    r"""

    2/4  3
     ____
    |\   |
    | \  |
    |  \ |
    |___\|

    0   1/5

    """
    assert hist.ndim == 1
    nsamples = len(hist)

    x0 = np.arange(nsamples)

    x = np.zeros(6 * nsamples)
    y = np.zeros(6 * nsamples)

    x[0::2] = np.repeat(x0, 3)
    x[1::2] = x[0::2] + 1

    y[2::6] = y[3::6] = y[4::6] = hist

    return np.c_[x, y]
