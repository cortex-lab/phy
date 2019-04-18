# -*- coding: utf-8 -*-

"""Test layout."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import product

import numpy as np
from numpy.testing import assert_equal as ae
from numpy.testing import assert_allclose as ac

from phy.utils import emit
from phy.utils.testing import _in_travis
from ..base import BaseVisual, BaseCanvas
from ..interact import Grid, Boxed, Stacked, Lasso
from ..panzoom import PanZoom
from ..transform import NDC
from ..visuals import ScatterVisual
from . import mouse_click


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

class MyTestVisual(BaseVisual):
    def __init__(self):
        super(MyTestVisual, self).__init__()
        self.vertex_shader = """
            attribute vec2 a_position;
            void main() {
                vec2 xy = a_position.xy;
                gl_Position = transform(xy);
                gl_PointSize = 10.;
            }
        """
        self.fragment_shader = """
            void main() {
                gl_FragColor = vec4(1, 1, 1, 1);
            }
        """
        self.set_primitive_type('points')

    def set_data(self):
        n = 1000
        self.n_vertices = 1000

        position = np.random.uniform(low=-1, high=+1, size=(n, 2))
        self.data = position
        self.program['a_position'] = position.astype(np.float32)

        emit('visual_set_data', self)


def _create_visual(qtbot, canvas, layout, box_index):
    c = canvas

    # Attach the layout *and* PanZoom. The order matters!
    layout.attach(c)
    PanZoom(aspect=None, constrain_bounds=NDC).attach(c)

    visual = MyTestVisual()
    c.add_visual(visual)
    visual.program['a_box_index'] = box_index.astype(np.float32)
    visual.set_data()

    c.show()
    qtbot.waitForWindowShown(c)


#------------------------------------------------------------------------------
# Test grid
#------------------------------------------------------------------------------

def test_grid_layout():
    grid = Grid((4, 8))
    ac(grid.map([0., 0.], (0, 0)), [[-0.875, 0.75]])
    ac(grid.map([0., 0.], (1, 3)), [[-0.125, 0.25]])
    ac(grid.map([0., 0.], (3, 7)), [[0.875, -0.75]])

    ac(grid.imap([[0.875, -0.75]], (3, 7)), [[0., 0.]])


def test_grid_closest_box():
    grid = Grid((3, 7))
    ac(grid.get_closest_box((0., 0.)), (1, 3))
    ac(grid.get_closest_box((-1., +1.)), (0, 0))
    ac(grid.get_closest_box((+1., -1.)), (2, 6))
    ac(grid.get_closest_box((-1., -1.)), (2, 0))
    ac(grid.get_closest_box((+1., +1.)), (0, 6))


def test_grid_1(qtbot, canvas):

    n = 1000

    box_index = [[i, j] for i, j in product(range(2), range(3))]
    box_index = np.repeat(box_index, n, axis=0)

    grid = Grid((2, 3))
    _create_visual(qtbot, canvas, grid, box_index)

    grid.add_boxes(canvas)

    # qtbot.stop()


def test_grid_2(qtbot, canvas):

    n = 1000

    box_index = [[i, j] for i, j in product(range(2), range(3))]
    box_index = np.repeat(box_index, n, axis=0)

    grid = Grid()
    _create_visual(qtbot, canvas, grid, box_index)
    grid.shape = (3, 3)
    assert grid.shape == (3, 3)

    # qtbot.stop()


#------------------------------------------------------------------------------
# Test boxed
#------------------------------------------------------------------------------

def test_boxed_1(qtbot, canvas):

    n = 6
    b = np.zeros((n, 4))

    b[:, 0] = b[:, 1] = np.linspace(-1., 1. - 1. / 3., n)
    b[:, 2] = b[:, 3] = np.linspace(-1. + 1. / 3., 1., n)

    n = 1000
    box_index = np.repeat(np.arange(6), n, axis=0)

    boxed = Boxed(box_bounds=b)
    _create_visual(qtbot, canvas, boxed, box_index)

    ae(boxed.box_bounds, b)
    boxed.box_bounds = b

    boxed.update_boxes(boxed.box_pos, boxed.box_size)
    ac(boxed.box_bounds, b * .9)

    # qtbot.stop()


def test_boxed_2(qtbot, canvas):
    """Test setting the box position and size dynamically."""

    n = 1000
    pos = np.c_[np.zeros(6), np.linspace(-1., 1., 6)]
    box_index = np.repeat(np.arange(6), n, axis=0)

    boxed = Boxed(box_pos=pos)
    _create_visual(qtbot, canvas, boxed, box_index)
    boxed.add_boxes(canvas)

    boxed.box_pos *= .25
    boxed.box_size = [1, .1]

    idx = boxed.get_closest_box((.5, .25))
    assert idx == 4

    # qtbot.stop()


def test_boxed_layout():

    n = 8
    b = np.zeros((n, 4))
    b[:, 0] = b[:, 1] = np.linspace(-1., 1. - 1. / 4., n)
    b[:, 2] = b[:, 3] = np.linspace(-1. + 1. / 4., 1., n)

    boxed = Boxed(box_bounds=b)
    ac(boxed.map([0., 0.], 0), [[-.875, -.875]])
    ac(boxed.map([0., 0.], 7), [[.875, .875]])
    ac(boxed.imap([[.875, .875]], 7), [[0., 0.]])


def test_boxed_closest_box():
    b = np.array([[-.5, -.5, 0., 0.],
                  [0., 0., +.5, +.5]])
    boxed = Boxed(box_bounds=b)

    ac(boxed.get_closest_box((-1, -1)), 0)
    ac(boxed.get_closest_box((-0.001, 0)), 0)
    ac(boxed.get_closest_box((+0.001, 0)), 1)
    ac(boxed.get_closest_box((-1, +1)), 0)


#------------------------------------------------------------------------------
# Test stacked
#------------------------------------------------------------------------------

def test_stacked_1(qtbot, canvas):

    n = 1000
    box_index = np.repeat(np.arange(6), n, axis=0)

    stacked = Stacked(n_boxes=6, origin='upper')
    _create_visual(qtbot, canvas, stacked, box_index)
    stacked.update_boxes(stacked.box_pos, stacked.box_size)

    # qtbot.stop()


def test_stacked_closest_box():
    stacked = Stacked(n_boxes=4, origin='upper')
    ac(stacked.get_closest_box((-.5, .9)), 0)
    ac(stacked.get_closest_box((+.5, -.9)), 3)

    stacked = Stacked(n_boxes=4, origin='lower')
    ac(stacked.get_closest_box((-.5, .9)), 3)
    ac(stacked.get_closest_box((+.5, -.9)), 0)


#------------------------------------------------------------------------------
# Test lasso
#------------------------------------------------------------------------------

def test_lasso_simple(qtbot):
    view = BaseCanvas()
    n = 1000

    x = .25 * np.random.randn(n)
    y = .25 * np.random.randn(n)

    scatter = ScatterVisual()
    view.add_visual(scatter)
    scatter.set_data(x=x, y=y)

    l = Lasso()
    l.attach(view)
    l.create_lasso_visual()

    view.show()
    #qtbot.waitForWindowShown(view)

    l.add((-.5, -.5))
    l.add((+.5, -.5))
    l.add((+.5, +.5))
    l.add((-.5, +.5))
    assert l.count == 4
    assert l.polygon.shape == (4, 2)
    b = [[-.5, -.5], [+.5, -.5], [+.5, +.5], [-.5, +.5]]
    ae(l.in_polygon(b), [False, False, True, True])
    assert str(l)

    # qtbot.stop()
    view.close()


def test_lasso_grid(qtbot, canvas):
    grid = Grid((1, 2))
    grid.attach(canvas)

    PanZoom(aspect=None).attach(canvas)
    grid.add_boxes(canvas)

    visual = MyTestVisual()
    canvas.add_visual(visual)
    # Right panel.
    box_index = np.zeros((1000, 2), dtype=np.float32)
    box_index[:, 1] = 1
    visual.program['a_box_index'] = box_index
    visual.set_data()

    # lasso interact
    l = Lasso()
    l.attach(canvas)
    l.create_lasso_visual()
    l.update_lasso_visual()

    canvas.show()
    qtbot.waitForWindowShown(canvas)
    qtbot.wait(10)

    def _ctrl_click(x, y, button='left'):
        mouse_click(qtbot, canvas, (x, y), button=button, modifiers=('Control',))

    # Square selection in the right panel.
    w, h = canvas.get_size()
    x0 = w / 2 + 100
    x1 = x0 + 200
    y0 = 100
    y1 = 300
    _ctrl_click(x0, y0)
    _ctrl_click(x1, y0)
    _ctrl_click(x1, y1)
    _ctrl_click(x0, y1)
    assert l.polygon.shape == (4, 2)
    assert l.box == (0, 1)

    inlasso = l.in_polygon(visual.data)
    assert .001 < inlasso.mean() < .999

    # Clear box.
    _ctrl_click(x0, y0, 'right')
    assert l.polygon.shape == (0, 2)
    assert l.box is None

    canvas.close()
