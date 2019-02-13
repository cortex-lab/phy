# -*- coding: utf-8 -*-

"""Test plot."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture, yield_fixture

from ..base import BaseCanvas
from ..panzoom import PanZoom


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

@yield_fixture
def canvas(qapp):
    c = BaseCanvas()
    yield c
    c.close()
    #del c


@fixture
def canvas_pz(canvas):
    pz = PanZoom(enable_mouse_wheel=True)
    pz.attach(canvas)
    return canvas
