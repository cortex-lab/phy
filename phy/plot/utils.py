# -*- coding: utf-8 -*-

"""Plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

import numpy as np
from vispy import gloo

from .transform import Range

logger = logging.getLogger(__name__)


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


def _get_texture(arr, default, n_items, from_bounds):
    """Prepare data to be uploaded as a texture, with casting to uint8.
    The from_bounds must be specified.
    """
    if not hasattr(default, '__len__'):  # pragma: no cover
        default = [default]
    n_cols = len(default)
    if arr is None:
        arr = np.tile(default, (n_items, 1))
    assert arr.shape == (n_items, n_cols)
    # Convert to 3D texture.
    arr = arr[np.newaxis, ...].astype(np.float32)
    assert arr.shape == (1, n_items, n_cols)
    # NOTE: we need to cast the texture to [0, 255] (uint8).
    # This is easy as soon as we assume that the signal bounds are in
    # [-1, 1].
    assert len(from_bounds) == 2
    m, M = map(float, from_bounds)
    assert np.all(arr >= m)
    assert np.all(arr <= M)
    arr = 255 * (arr - m) / (M - m)
    assert np.all(arr >= 0)
    assert np.all(arr <= 255)
    arr = arr.astype(np.uint8)
    return arr


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


def _get_boxes(pos, margin=0):
    """Generate non-overlapping boxes in NDC from a set of positions."""

    # Get x, y.
    pos = np.asarray(pos, dtype=np.float32)
    x, y = pos.T
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    # Deal with degenerate x case.
    xmin, xmax = x.min(), x.max()
    if xmin == xmax:
        wmax = 1.
    else:
        wmax = xmax - xmin

    ar = .5

    def f1(w):
        """Return true if the configuration with the current box size
        is non-overlapping."""
        h = w * ar  # fixed aspect ratio
        return not _boxes_overlap(x - w, y - h, x + w, y + h)

    # Find the largest box size leading to non-overlapping boxes.
    w = _binary_search(f1, 0, wmax)
    w = w * (1 - margin)  # margin
    h = w * ar  # aspect ratio

    x0, y0 = x - w, y - h
    x1, y1 = x + w, y + h

    # Renormalize the whole thing by keeping the aspect ratio.
    x0min, y0min, x1max, y1max = x0.min(), y0.min(), x1.max(), y1.max()
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
