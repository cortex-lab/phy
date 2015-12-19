# -*- coding: utf-8 -*-

"""Test VisPy."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from vispy.app import use_app
from pytest import yield_fixture

from ..base import BaseCanvas
from ..panzoom import PanZoom


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

@yield_fixture
def canvas(qapp):
    use_app('pyqt4')
    c = BaseCanvas(keys='interactive')
    yield c
    c.close()


@yield_fixture
def canvas_pz(canvas):
    PanZoom(enable_mouse_wheel=True).attach(canvas)
    yield canvas
