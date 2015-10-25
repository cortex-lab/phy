# -*- coding: utf-8 -*-

"""Plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op

import numpy as np

from vispy import gloo, config

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Misc
#------------------------------------------------------------------------------

def _load_shader(filename):
    """Load a shader file."""
    curdir = op.dirname(op.realpath(__file__))
    glsl_path = op.join(curdir, 'glsl')
    if not config['include_path']:
        config['include_path'] = [glsl_path]
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
