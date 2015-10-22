# -*- coding: utf-8 -*-

"""Test panzoom."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..panzoom import PanZoom


#------------------------------------------------------------------------------
# Test panzoom
#------------------------------------------------------------------------------

def test_panzoom_basic_attrs():
    panzoom = PanZoom()

    assert not panzoom.is_attached()

    # Aspect.
    assert panzoom.aspect == 1.
    panzoom.aspect = 2.
    assert panzoom.aspect == 2.

    # Constraints.
    for name in ('xmin', 'xmax', 'ymin', 'ymax'):
        assert getattr(panzoom, name) is None
        setattr(panzoom, name, 1.)
        assert getattr(panzoom, name) == 1.

    for name, v in (('zmin', 1e-5), ('zmax', 1e5)):
        assert getattr(panzoom, name) == v
        setattr(panzoom, name, v * 2)
        assert getattr(panzoom, name) == v * 2

    assert list(panzoom.iter_attached_visuals()) == []


def test_panzoom_basic_pan_zoom():
    panzoom = PanZoom()

    # Pan.
    assert panzoom.pan == [0., 0.]
    panzoom.pan = (1., -1.)
    assert panzoom.pan == [1., -1.]

    # Zoom.
    assert panzoom.zoom == [1., 1.]
    panzoom.zoom = (2., .5)
    assert panzoom.zoom == [2., .5]
    panzoom.zoom = (1., 1.)

    # Pan delta.
    panzoom.pan_delta((-1., 1.))
    assert panzoom.pan == [0., 0.]

    # Zoom delta.
    panzoom.zoom_delta((1., 1.))
    assert panzoom.zoom[0] > 2
    assert panzoom.zoom[0] == panzoom.zoom[1]
    panzoom.zoom = (1., 1.)

    # Zoom delta.
    panzoom.zoom_delta((2., 3.), (.5, .5))
    assert panzoom.zoom[0] > 2
    assert panzoom.zoom[1] > 3 * panzoom.zoom[0]
