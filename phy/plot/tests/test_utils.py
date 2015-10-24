# -*- coding: utf-8 -*-

"""Test plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op

import numpy as np
from numpy.testing import assert_allclose as ac
from vispy import config

from ..utils import (_load_shader,
                     _create_program,
                     _tesselate_histogram,
                     _enable_depth_mask,
                     )


#------------------------------------------------------------------------------
# Test utilities
#------------------------------------------------------------------------------

def test_load_shader():
    assert 'main()' in _load_shader('ax.vert')
    assert config['include_path']
    assert op.exists(config['include_path'][0])
    assert op.isdir(config['include_path'][0])
    assert os.listdir(config['include_path'][0])


def test_create_program():
    p = _create_program('box')
    assert p.shaders[0]
    assert p.shaders[1]


def test_tesselate_histogram():
    n = 7
    hist = np.arange(n)
    thist = _tesselate_histogram(hist)
    assert thist.shape == (6 * n, 2)
    ac(thist[0], [0, 0])
    ac(thist[-3], [n, n - 1])
    ac(thist[-1], [n, 0])


def test_enable_depth_mask(qtbot, canvas):

    @canvas.connect
    def on_draw(e):
        _enable_depth_mask()

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)
