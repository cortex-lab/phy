# -*- coding: utf-8 -*-

"""Test grid."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import product

import numpy as np
from vispy.util import keys
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
            gl_PointSize = 2.;
        }
        """
    fragment = """
        void main() {
            gl_FragColor = vec4(1, 1, 1, 1);
        }
    """
    gl_primitive_type = 'points'

    def __init__(self):
        super(MyTestVisual, self).__init__()
        self.transforms = [GPU()]
        self.set_data()

    def set_data(self):
        n = 1000

        box = [[i, j] for i, j in product(range(2), range(3))]
        box = np.repeat(box, n, axis=0)

        coeff = [(1 + i + j) for i, j in product(range(2), range(3))]
        coeff = np.repeat(coeff, n)
        coeff = coeff[:, None]

        position = .1 * coeff * np.random.randn(2 * 3 * n, 2)

        self.data['a_position'] = position.astype(np.float32)
        self.data['a_box'] = box.astype(np.float32)


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

def test_grid_1(qtbot, canvas, visual, grid):

    # Zoom with the keyboard.
    canvas.events.key_press(key=keys.Key('+'))
    assert grid.zoom > 1

    # Unzoom with the keyboard.
    canvas.events.key_press(key=keys.Key('-'))
    assert grid.zoom == 1.

    # Set the zoom explicitly.
    grid.zoom = 2
    assert grid.zoom == 2.

    # No effect with modifiers.
    canvas.events.key_press(key=keys.Key('r'), modifiers=(keys.CONTROL,))
    assert grid.zoom == 2.

    # Press 'R'.
    canvas.events.key_press(key=keys.Key('r'))
    assert grid.zoom == 1.
