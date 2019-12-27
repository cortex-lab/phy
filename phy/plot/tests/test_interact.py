# -*- coding: utf-8 -*-

"""Test layout."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import product

import numpy as np
from numpy.testing import assert_equal as ae
from numpy.testing import assert_allclose as ac

from ..base import BaseVisual, BaseCanvas
from ..interact import Grid, Boxed, Stacked, Lasso
from ..panzoom import PanZoom
from ..transform import NDC
from ..visuals import ScatterVisual
from . import mouse_click


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

N = 10000


class MyTestVisual(BaseVisual):
    def __init__(self):
        super(MyTestVisual, self).__init__()
        self.vertex_shader = """
            attribute vec2 a_position;
            void main() {
                vec2 xy = a_position.xy;
                gl_Position = transform(xy);
                gl_PointSize = 5.;
            }
        """
        self.fragment_shader = """
            void main() {
                gl_FragColor = vec4(1, 1, 1, 1);
            }
        """
        self.set_primitive_type('points')

    def set_data(self):
        self.n_vertices = N
        position = np.random.uniform(low=-1, high=+1, size=(N, 2))
        self.data = position
        self.program['a_position'] = position.astype(np.float32)

        self.emit_visual_set_data()


def _create_visual(qtbot, canvas, layout, box_index):
    c = canvas

    # Attach the layout *and* PanZoom. The order matters!
    layout.attach(c)
    PanZoom(aspect=None, constrain_bounds=NDC).attach(c)

    visual = MyTestVisual()
    c.add_visual(visual)
    visual.set_data()
    visual.set_box_index(box_index)

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

    n = N // 10

    box_index = [[i, j] for i, j in product(range(2), range(5))]
    box_index = np.repeat(box_index, n, axis=0)

    grid = Grid((2, 5))
    _create_visual(qtbot, canvas, grid, box_index)

    grid.add_boxes(canvas)

    # qtbot.stop()


def test_grid_2(qtbot, canvas):

    n = N // 10

    box_index = [[i, j] for i, j in product(range(2), range(5))]
    box_index = np.repeat(box_index, n, axis=0)

    grid = Grid()
    _create_visual(qtbot, canvas, grid, box_index)
    grid.shape = (5, 2)
    assert grid.shape == (5, 2)

    grid.scaling = (.5, 2)
    assert grid.scaling == (.5, 2)

    # qtbot.stop()


#------------------------------------------------------------------------------
# Test boxed
#------------------------------------------------------------------------------

def test_boxed_1(qtbot, canvas):

    n = 10
    b = np.zeros((n, 2))
    b[:, 1] = np.linspace(-1., 1., n)

    box_index = np.repeat(np.arange(n), N // n, axis=0)
    assert box_index.shape == (N,)

    boxed = Boxed(box_pos=b)
    _create_visual(qtbot, canvas, boxed, box_index)
    boxed.add_boxes(canvas)

    assert boxed.box_scaling == (1, 1)
    assert boxed.layout_scaling == (1, 1)

    ac(boxed.box_pos[:, 0], 0, atol=1e-9)
    assert boxed.box_size[0] >= .9
    assert boxed.box_size[1] >= .05

    assert boxed.box_bounds.shape == (n, 4)

    boxed.expand_box_width()
    boxed.shrink_box_width()
    boxed.expand_box_height()
    boxed.shrink_box_height()
    boxed.expand_layout_width()
    boxed.shrink_layout_width()
    boxed.expand_layout_height()
    boxed.shrink_layout_height()

    # qtbot.stop()


def test_boxed_2(qtbot, canvas):
    from ..visuals import PlotAggVisual

    n = 10
    b = np.zeros((n, 2))
    b[:, 1] = np.linspace(-1., 1., n)

    box_index = np.repeat(np.arange(n), 2 * (N + 2), axis=0)

    boxed = Boxed(box_pos=b)
    c = canvas
    boxed.attach(c)
    PanZoom(aspect=None, constrain_bounds=NDC).attach(c)

    t = np.linspace(-1, 1, N)
    x = np.atleast_2d(t)
    y = np.atleast_2d(.5 * np.sin(20 * t))

    x = np.tile(x, (n, 1))
    y = np.tile(y, (n, 1))

    visual = PlotAggVisual()
    c.add_visual(visual)
    visual.set_data(x=x, y=y)
    visual.set_box_index(box_index)

    c.show()
    qtbot.waitForWindowShown(c)


#------------------------------------------------------------------------------
# Test stacked
#------------------------------------------------------------------------------

def test_stacked_1(qtbot, canvas):

    n = 10
    box_index = np.repeat(np.arange(n), N // n, axis=0)

    stacked = Stacked(n_boxes=n, origin='top')
    _create_visual(qtbot, canvas, stacked, box_index)

    assert stacked.origin == 'top'

    stacked.origin = 'bottom'
    assert stacked.origin == 'bottom'

    # qtbot.stop()


def test_stacked_closest_box():
    stacked = Stacked(n_boxes=4, origin='top')
    ac(stacked.get_closest_box((-.5, .9)), 0)
    ac(stacked.get_closest_box((+.5, -.9)), 3)

    stacked = Stacked(n_boxes=4, origin='bottom')
    ac(stacked.get_closest_box((-.5, .9)), 3)
    ac(stacked.get_closest_box((+.5, -.9)), 0)

    stacked.n_boxes = 3


#------------------------------------------------------------------------------
# Test lasso
#------------------------------------------------------------------------------

def test_lasso_simple(qtbot):
    view = BaseCanvas()

    x = .25 * np.random.randn(N)
    y = .25 * np.random.randn(N)

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
    box_index = np.zeros((N, 2), dtype=np.float32)
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
    qtbot.wait(20)

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

    qtbot.wait(20)
    canvas.close()
