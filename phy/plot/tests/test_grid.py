# -*- coding: utf-8 -*-

"""Test grid."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from pytest import yield_fixture

from ..base import BaseVisual
from ..grid import Grid
from ..transform import GPU


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

class MyTestVisual(BaseVisual):
    vertex = """
        attribute vec2 a_position;
        void main() {
            gl_Position = transform(a_position);
        }
        """
    fragment = """
        void main() {
            gl_FragColor = vec4(1, 1, 1, 1);
        }
    """
    gl_primitive_type = 'lines'

    def __init__(self):
        super(MyTestVisual, self).__init__()
        self.transforms = [GPU()]
        self.set_data()

    def set_data(self):
        self.data['a_position'] = [[-1, 0], [1, 0]]


@yield_fixture
def visual():
    yield MyTestVisual()


@yield_fixture
def grid(qtbot, canvas, visual):
    visual.attach(canvas, 'Grid')

    grid = Grid(shape=(2, 3))
    grid.attach(canvas)

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)

    yield grid


#------------------------------------------------------------------------------
# Test grid
#------------------------------------------------------------------------------

def test_grid_1(qtbot, visual, grid):
    qtbot.stop()
