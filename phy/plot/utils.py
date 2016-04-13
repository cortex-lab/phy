# -*- coding: utf-8 -*-

"""Plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

import numpy as np
from six import string_types
from vispy import gloo

from .transform import Range, NDC

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Box positioning
#------------------------------------------------------------------------------

def _boxes_overlap(x0, y0, x1, y1):
    n = len(x0)
    overlap_matrix = ((x0 < x1.T) & (x1 > x0.T) & (y0 < y1.T) & (y1 > y0.T))
    overlap_matrix[np.arange(n), np.arange(n)] = False
    return np.any(overlap_matrix.ravel())


def _binary_search(f, xmin, xmax, eps=1e-9):
    """Return the largest x such f(x) is True."""
    middle = (xmax + xmin) / 2.
    while xmax - xmin > eps:
        assert xmin < xmax
        middle = (xmax + xmin) / 2.
        if f(xmax):
            return xmax
        if not f(xmin):
            return xmin
        if f(middle):
            xmin = middle
        else:
            xmax = middle
    return middle


def _get_box_size(x, y, ar=.5, margin=0):
    logger.debug("Get box size for %d points.", len(x))
    # Deal with degenerate x case.
    xmin, xmax = x.min(), x.max()
    if xmin == xmax:
        # If all positions are vertical, the width can be maximum.
        wmax = 1.
    else:
        wmax = xmax - xmin

    def f1(w):
        """Return true if the configuration with the current box size
        is non-overlapping."""
        # NOTE: w|h are the *half* width|height.
        h = w * ar  # fixed aspect ratio
        return not _boxes_overlap(x - w, y - h, x + w, y + h)

    # Find the largest box size leading to non-overlapping boxes.
    w = _binary_search(f1, 0, wmax)
    w = w * (1 - margin)  # margin
    # Clip the half-width.
    h = w * ar  # aspect ratio

    return w, h


def _get_boxes(pos, size=None, margin=0, keep_aspect_ratio=True):
    """Generate non-overlapping boxes in NDC from a set of positions."""

    # Get x, y.
    pos = np.asarray(pos, dtype=np.float64)
    x, y = pos.T
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    w, h = size if size is not None else _get_box_size(x, y, margin=margin)

    x0, y0 = x - w, y - h
    x1, y1 = x + w, y + h

    # Renormalize the whole thing by keeping the aspect ratio.
    x0min, y0min, x1max, y1max = x0.min(), y0.min(), x1.max(), y1.max()
    if not keep_aspect_ratio:
        b = (x0min, y0min, x1max, y1max)
    else:
        dx = x1max - x0min
        dy = y1max - y0min
        if dx > dy:
            b = (x0min, (y1max + y0min) / 2. - dx / 2.,
                 x1max, (y1max + y0min) / 2. + dx / 2.)
        else:
            b = ((x1max + x0min) / 2. - dy / 2., y0min,
                 (x1max + x0min) / 2. + dy / 2., y1max)
    r = Range(from_bounds=b,
              to_bounds=(-1, -1, 1, 1))
    return np.c_[r.apply(np.c_[x0, y0]), r.apply(np.c_[x1, y1])]


def _get_box_pos_size(box_bounds):
    box_bounds = np.asarray(box_bounds)
    x0, y0, x1, y1 = box_bounds.T
    w = (x1 - x0) * .5
    h = (y1 - y0) * .5
    x = (x0 + x1) * .5
    y = (y0 + y1) * .5
    return np.c_[x, y], (w.mean(), h.mean())


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


def _check_data_bounds(data_bounds):
    assert data_bounds.ndim == 2
    assert data_bounds.shape[1] == 4
    assert np.all(data_bounds[:, 0] < data_bounds[:, 2])
    assert np.all(data_bounds[:, 1] < data_bounds[:, 3])


def _get_data_bounds(data_bounds, pos=None, length=None):
    """"Prepare data bounds, possibly using min/max of the data."""
    if data_bounds is None or (isinstance(data_bounds, string_types) and
                               data_bounds == 'auto'):
        if pos is not None and len(pos):
            m, M = pos.min(axis=0), pos.max(axis=0)
            data_bounds = [m[0], m[1], M[0], M[1]]
        else:
            data_bounds = NDC
    data_bounds = np.atleast_2d(data_bounds)

    ind_x = data_bounds[:, 0] == data_bounds[:, 2]
    ind_y = data_bounds[:, 1] == data_bounds[:, 3]
    if np.sum(ind_x):
        data_bounds[ind_x, 0] -= 1
        data_bounds[ind_x, 2] += 1
    if np.sum(ind_y):
        data_bounds[ind_y, 1] -= 1
        data_bounds[ind_y, 3] += 1

    # Extend the data_bounds if needed.
    if length is None:
        length = pos.shape[0] if pos is not None else 1
    if data_bounds.shape[0] == 1:
        data_bounds = np.tile(data_bounds, (length, 1))

    # Check the shape of data_bounds.
    assert data_bounds.shape == (length, 4)

    _check_data_bounds(data_bounds)
    return data_bounds


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


#------------------------------------------------------------------------------
# Misc
#------------------------------------------------------------------------------

def _load_shader(filename):
    """Load a shader file."""
    curdir = op.dirname(op.realpath(__file__))
    glsl_path = op.join(curdir, 'glsl')
    path = op.join(glsl_path, filename)
    with open(path, 'r') as f:
        return f.read()


def _tesselate_histogram(hist):
    """

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


def _enable_depth_mask():
    gloo.set_state(clear_color='black',
                   depth_test=True,
                   depth_range=(0., 1.),
                   # depth_mask='true',
                   depth_func='lequal',
                   blend=True,
                   blend_func=('src_alpha', 'one_minus_src_alpha'))
    gloo.set_clear_depth(1.0)
