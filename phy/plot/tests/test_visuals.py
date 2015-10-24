# -*- coding: utf-8 -*-

"""Test visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from pytest import mark

from ..visuals import ScatterVisual


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Test visuals
#------------------------------------------------------------------------------

def test_scatter_empty(qtbot, canvas):

    v = ScatterVisual()
    v.attach(canvas)

    n = 0
    pos = np.zeros((n, 2))
    v.set_data(pos=pos)

    canvas.show()
    qtbot.stop()


@mark.parametrize('marker_type', ScatterVisual._supported_marker_types)
def test_scatter_markers(qtbot, canvas_pz, marker_type):

    # Try all marker types.
    v = ScatterVisual(marker_type=marker_type)
    v.attach(canvas_pz)

    n = 100
    pos = .2 * np.random.randn(n, 2)
    v.set_data(pos=pos)

    canvas_pz.show()
    # qtbot.stop()


def test_scatter_custom(qtbot, canvas_pz):

    v = ScatterVisual()
    v.attach(canvas_pz)

    n = 100

    # Random position.
    pos = .2 * np.random.randn(n, 2)

    # Random colors.
    c = np.random.uniform(.4, .7, size=(n, 4))
    c[:, -1] = .5

    # Random sizes
    s = 5 + 20 * np.random.rand(n)

    v.set_data(pos=pos, colors=c, size=s)

    canvas_pz.show()
    # qtbot.stop()
