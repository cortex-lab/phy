# -*- coding: utf-8 -*-

"""Test plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from vispy import gloo

from ..utils import (_load_shader,
                     _tesselate_histogram,
                     _enable_depth_mask,
                     )


#------------------------------------------------------------------------------
# Test utilities
#------------------------------------------------------------------------------

def test_load_shader():
    assert 'main()' in _load_shader('ax.vert')


def test_tesselate_histogram():
    n = 5
    hist = np.arange(n)
    thist = _tesselate_histogram(hist)
    assert thist.shape == (5 * n + 1, 2)
    ae(thist[0], [-1, -1])
    ae(thist[-1], [1, -1])


def test_enable_depth_mask(qtbot, canvas):

    @canvas.connect
    def on_draw(e):
        _enable_depth_mask()

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()
