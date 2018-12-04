# -*- coding: utf-8 -*-

"""Test plot."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

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
    del c


@yield_fixture
def canvas_pz(canvas):
    PanZoom(enable_mouse_wheel=True).attach(canvas)
    yield canvas
