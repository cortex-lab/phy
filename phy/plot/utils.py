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
    assert hist.ndim == 1
    nsamples = len(hist)
    dx = 2. / nsamples

    x0 = -1 + dx * np.arange(nsamples)

    x = np.zeros(5 * nsamples + 1)
    y = -1.0 * np.ones(5 * nsamples + 1)

    x[0:-1:5] = x0
    x[1::5] = x0
    x[2::5] = x0 + dx
    x[3::5] = x0
    x[4::5] = x0 + dx
    x[-1] = 1

    y[1::5] = y[2::5] = -1 + 2. * hist

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
