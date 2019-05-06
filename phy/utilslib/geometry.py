# -*- coding: utf-8 -*-

"""Plotting utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
from six import string_types

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Common probe layouts
#------------------------------------------------------------------------------

def linear_positions(n_channels):
    """Linear channel positions along the vertical axis."""
    return np.c_[np.zeros(n_channels),
                 np.linspace(0., 1., n_channels)]


def staggered_positions(n_channels):
    """Generate channel positions for a staggered probe."""
    i = np.arange(n_channels - 1)
    x, y = (-1) ** i * (5 + i), 10 * (i + 1)
    pos = np.flipud(np.r_[np.zeros((1, 2)), np.c_[x, y]])
    return pos


#------------------------------------------------------------------------------
# Box positioning
#------------------------------------------------------------------------------

def range_transform(from_bounds, to_bounds, positions):
    from_bounds = np.asarray(from_bounds)
    to_bounds = np.asarray(to_bounds)
    positions = np.asarray(positions)

    f0 = from_bounds[..., :2]
    f1 = from_bounds[..., 2:]
    t0 = to_bounds[..., :2]
    t1 = to_bounds[..., 2:]

    d = (f1 - f0)
    d[d == 0] = 1

    out = positions.copy()
    out -= f0
    out *= (t1 - t0) / d
    out += t0
    return out


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
    logger.log(5, "Get box size for %d points.", len(x))
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

    # Margin.
    a = margin

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
    fb = np.asarray(b)
    tb = np.asarray((-1 + a, -1 + a, 1 - a, 1 - a))
    return np.c_[
        range_transform(fb, tb, np.c_[x0, y0]),
        range_transform(fb, tb, np.c_[x1, y1])
    ]


def _get_box_pos_size(box_bounds):
    box_bounds = np.asarray(box_bounds)
    x0, y0, x1, y1 = box_bounds.T
    w = (x1 - x0) * .5
    h = (y1 - y0) * .5
    x = (x0 + x1) * .5
    y = (y0 + y1) * .5
    return np.c_[x, y], (w.mean(), h.mean())


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
            data_bounds = [-1, -1, 1, 1]
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
