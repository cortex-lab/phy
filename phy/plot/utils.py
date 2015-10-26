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


def _rescale_positions(pos, size):
    """Rescale positions so that the boxes fit in NDC."""
    a, b = size

    # Get x, y.
    pos = np.asarray(pos, dtype=np.float32)
    x, y = pos.T
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Renormalize into [-1, 1].
    pos = Range(from_bounds=(xmin, ymin, xmax, ymax),
                to_bounds=(-1, -1, 1, 1)).apply(pos)

    # Rescale the positions so that everything fits in the box.
    alpha = 1.
    if xmin != 0:
        alpha = min(alpha, (-1 + a) / xmin)
    if xmax != 0:
        alpha = min(alpha, (+1 - a) / xmax)

    beta = 1.
    if ymin != 0:
        beta = min(beta, (-1 + b) / ymin)
    if ymax != 0:
        beta = min(beta, (+1 - b) / ymax)

    # Get xy01.
    x0, y0 = alpha * x - a, beta * y - b
    x1, y1 = alpha * x + a, beta * y + b

    return x0, y0, x1, y1


def _get_boxes(pos):
    """Generate non-overlapping boxes in NDC from a set of positions."""

    # Find a box_size such that the boxes are non-overlapping.
    def f(size):
        a, b = size
        x0, y0, x1, y1 = _rescale_positions(pos, size)

        if _boxes_overlap(x0, y0, x1, y1):
            return 0.

        return -(2 * a + b)

    cons = [{'type': 'ineq', 'fun': lambda s: s[0]},
            {'type': 'ineq', 'fun': lambda s: s[1]},
            {'type': 'ineq', 'fun': lambda s: 1 - s[0]},
            {'type': 'ineq', 'fun': lambda s: 1 - s[1]},
            ]

    from scipy.optimize import minimize
    res = minimize(f, (.05, .01),
                   constraints=cons,
                   )
    w, h = res.x
    assert f((w, h)) < 0

    return _rescale_positions(pos, (w, h))
