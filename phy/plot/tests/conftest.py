# -*- coding: utf-8 -*-

"""Test plot."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture

from ..base import BaseCanvas
from ..panzoom import PanZoom


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

@fixture
def canvas(qapp, qtbot):
    c = BaseCanvas()
    yield c
    c.close()
    del c
    qtbot.wait(50)


@fixture
def canvas_pz(canvas):
    pz = PanZoom(enable_mouse_wheel=True)
    pz.attach(canvas)
    return canvas
