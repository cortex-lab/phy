# -*- coding: utf-8 -*-

"""Test interact."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import product

import numpy as np
from vispy.util import keys
from pytest import yield_fixture

from ..base import BaseVisual, BaseCanvas
from ..interact import Grid, Boxed


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

    def get_shaders(self):
        return self.vertex, self.fragment

    def set_data(self):
        n = 1000

        coeff = [(1 + i + j) for i, j in product(range(2), range(3))]
        coeff = np.repeat(coeff, n)
        coeff = coeff[:, None]

        position = .1 * coeff * np.random.randn(2 * 3 * n, 2)

        self.program['a_position'] = position.astype(np.float32)


@yield_fixture
def canvas_grid(qapp):
    c = BaseCanvas(keys='interactive', interact=Grid(shape=(2, 3)))
    yield c
    c.close()


@yield_fixture
def canvas_boxed(qapp):
    n = 6
    b = np.zeros((n, 4))

    b[:, 0] = b[:, 1] = np.linspace(-1., 1. - 1. / 3., n)
    b[:, 2] = b[:, 3] = np.linspace(-1. + 1. / 3., 1., n)

    c = BaseCanvas(keys='interactive', interact=Boxed(box_bounds=b))
    yield c
    c.close()


def get_interact(qtbot, canvas, box_index):
    c = canvas

    visual = MyTestVisual()
    visual.attach(c)
    visual.set_data()

    visual.program['a_box_index'] = box_index.astype(np.float32)

    c.show()
    qtbot.waitForWindowShown(c.native)

    return c.interact


#------------------------------------------------------------------------------
# Test grid
#------------------------------------------------------------------------------

def test_grid_1(qtbot, canvas_grid):
    c = canvas_grid
    n = 1000

    box_index = [[i, j] for i, j in product(range(2), range(3))]
    box_index = np.repeat(box_index, n, axis=0)

    grid = get_interact(qtbot, canvas_grid, box_index)

    # Zoom with the keyboard.
    c.events.key_press(key=keys.Key('+'))
    assert grid.zoom > 1

    # Unzoom with the keyboard.
    c.events.key_press(key=keys.Key('-'))
    assert grid.zoom == 1.

    # Set the zoom explicitly.
    grid.zoom = 2
    assert grid.zoom == 2.

    # No effect with modifiers.
    c.events.key_press(key=keys.Key('r'), modifiers=(keys.CONTROL,))
    assert grid.zoom == 2.

    # Press 'R'.
    c.events.key_press(key=keys.Key('r'))
    assert grid.zoom == 1.

    qtbot.stop()


def test_boxed_1(qtbot, canvas_boxed):
    c = canvas_boxed

    n = 1000
    box_index = np.repeat(np.arange(6), n, axis=0)

    boxed = get_interact(qtbot, canvas_boxed, box_index)

    qtbot.stop()
