# -*- coding: utf-8 -*-

"""Test panzoom."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

from numpy.testing import assert_allclose as ac
from pytest import fixture

from . import mouse_drag, key_press
from ..base import BaseVisual
from ..panzoom import PanZoom


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

class MyTestVisual(BaseVisual):
    def __init__(self):
        super(MyTestVisual, self).__init__()
        self.set_shader('simple')
        self.set_primitive_type('lines')

    def set_data(self):
        self.n_vertices = 2
        self.program['a_position'] = [[-1, 0], [1, 0]]
        self.program['u_color'] = [1, 1, 1, 1]
        self.emit_visual_set_data()


@fixture
def panzoom(qtbot, canvas_pz):
    c = canvas_pz
    visual = MyTestVisual()
    c.add_visual(visual)
    visual.set_data()

    c.show()
    qtbot.waitForWindowShown(c)

    yield c.panzoom

    if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
        qtbot.stop()
    c.close()


#------------------------------------------------------------------------------
# Test panzoom
#------------------------------------------------------------------------------

def test_panzoom_basic_attrs():
    pz = PanZoom()

    # Aspect.
    assert pz.aspect is None
    pz.aspect = 2.
    assert pz.aspect == 2.

    # Constraints.
    for name in ('xmin', 'xmax', 'ymin', 'ymax'):
        assert getattr(pz, name) is None
        setattr(pz, name, 1.)
        assert getattr(pz, name) == 1.

    for name, v in (('zmin', 1e-5), ('zmax', 1e5)):
        assert getattr(pz, name) == v
        setattr(pz, name, v * 2)
        assert getattr(pz, name) == v * 2


def test_panzoom_basic_constrain():
    pz = PanZoom(constrain_bounds=(-1, -1, 1, 10))
    pz.set_constrain_bounds((-1, -1, 1, 1))

    # Aspect.
    assert pz.aspect is None
    pz.aspect = 2.
    assert pz.aspect == 2.

    # Constraints.
    assert pz.xmin == pz.ymin == -1
    assert pz.xmax == pz.ymax == +1


def test_panzoom_basic_pan_zoom():
    pz = PanZoom()

    # Pan.
    assert pz.pan == [0., 0.]
    pz.pan = (1., -1.)
    assert pz.pan == [1., -1.]

    # Zoom.
    assert pz.zoom == [1., 1.]
    pz.zoom = (2., .5)
    assert pz.zoom == [2., .5]
    pz.zoom = (1., 1.)

    # Pan delta.
    pz.pan_delta((-1., 1.))
    assert pz.pan == [0., 0.]

    # Zoom delta.
    pz.zoom_delta((1., 1.))
    assert pz.zoom[0] > 2
    assert pz.zoom[0] == pz.zoom[1]
    pz.zoom = (1., 1.)

    # Zoom delta.
    pz.zoom_delta((2., 3.), (.5, .5))
    assert pz.zoom[0] > 2
    assert pz.zoom[1] > 3 * pz.zoom[0]


def test_panzoom_map():
    pz = PanZoom()
    pz.pan = (1., -1.)
    ac(pz.map([0., 0.]), [[1., -1.]])

    pz.zoom = (2., .5)
    ac(pz.map([0., 0.]), [[2., -.5]])

    ac(pz.imap([2., -.5]), [[0., 0.]])


def test_panzoom_constraints_x():
    pz = PanZoom()
    pz.xmin, pz.xmax = -2, 2

    # Pan beyond the bounds.
    pz.pan_delta((-2, 2))
    assert pz.pan == [-1, 2]
    pz.reset()

    # Zoom beyond the bounds.
    pz.zoom_delta((-1, -2))
    assert pz.pan == [0, 0]
    assert pz.zoom[0] == .5
    assert pz.zoom[1] < .5


def test_panzoom_constraints_y():
    pz = PanZoom()
    pz.ymin, pz.ymax = -2, 2

    # Pan beyond the bounds.
    pz.pan_delta((2, -2))
    assert pz.pan == [2, -1]
    pz.reset()

    # Zoom beyond the bounds.
    pz.zoom_delta((-2, -1))
    assert pz.pan == [0, 0]
    assert pz.zoom[0] < .5
    assert pz.zoom[1] == .5


def test_panzoom_constraints_z():
    pz = PanZoom()
    pz.zmin, pz.zmax = .5, 2

    # Zoom beyond the bounds.
    pz.zoom_delta((-10, -10))
    assert pz.zoom == [.5, .5]
    pz.reset()

    pz.zoom_delta((10, 10))
    assert pz.zoom == [2, 2]


def test_panzoom_set_range():
    pz = PanZoom()

    def _test_range(*bounds):
        pz.set_range(bounds)
        ac(pz.get_range(), bounds)

    _test_range(-1, -1, 1, 1)
    ac(pz.zoom, (1, 1))

    _test_range(-.5, -.5, .5, .5)
    ac(pz.zoom, (2, 2))

    _test_range(0, 0, 1, 1)
    ac(pz.zoom, (2, 2))

    _test_range(-1, 0, 1, 1)
    ac(pz.zoom, (1, 2))

    pz.set_range((-1, 0, 1, 1), keep_aspect=True)
    ac(pz.zoom, (1, 1))


def test_panzoom_mouse_pos():
    pz = PanZoom()
    pz.zoom_delta((10, 10), (.5, .25))
    pos = pz.window_to_ndc((.01, -.01))
    ac(pos, (.5, .25), atol=1e-3)


#------------------------------------------------------------------------------
# Test panzoom on canvas
#------------------------------------------------------------------------------

def test_panzoom_pan_mouse(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Pan with mouse.
    mouse_drag(qtbot, c, (100, 0), (200, 0))
    assert pz.pan[0] > 0
    assert pz.pan[1] == 0
    pz.pan = (0, 0)

    # Panning with a modifier should not pan.
    mouse_drag(qtbot, c, (100, 0), (200, 0), modifiers=('Control',))
    assert pz.pan == [0, 0]


def test_panzoom_pan_keyboard(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Pan with keyboard.
    key_press(qtbot, c, 'Up')
    assert pz.pan[0] == 0
    assert pz.pan[1] < 0

    # All panning movements with keys.
    key_press(qtbot, c, 'Left')
    key_press(qtbot, c, 'Down')
    key_press(qtbot, c, 'Right')
    assert pz.pan == [0, 0]

    # Reset.
    key_press(qtbot, c, 'Right')
    key_press(qtbot, c, 'R')
    pz.reset()
    assert pz.pan == [0, 0]

    # Using modifiers should not pan.
    key_press(qtbot, c, 'Up', modifiers=('Control',))
    assert pz.pan == [0, 0]

    # Disable keyboard pan.
    pz.enable_keyboard_pan = False
    key_press(qtbot, c, 'Up', modifiers=('Control',))
    assert pz.pan == [0, 0]


def test_panzoom_zoom_mouse(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Zoom with mouse.
    mouse_drag(qtbot, c, (10, 10), (5, 5), button='right')
    assert pz.pan[0] < 0
    assert pz.pan[1] < 0
    assert pz.zoom[0] < 1
    assert pz.zoom[1] > 1
    pz.reset()

    pz.aspect = 1

    mouse_drag(qtbot, c, (10, 10), (5, 100), button='right')
    assert pz.zoom[0] < 1
    assert pz.zoom[1] < 1

    mouse_drag(qtbot, c, (10, 10), (100, 5), button='right')
    assert pz.zoom[0] < 1
    assert pz.zoom[1] < 1

    mouse_drag(qtbot, c, (10, 10), (-5, -100), button='right')
    assert pz.zoom[0] > 1
    assert pz.zoom[1] > 1


def test_panzoom_zoom_keyboard(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Zoom with keyboard.
    key_press(qtbot, c, 'Plus')
    assert pz.pan == [0, 0]
    assert pz.zoom[0] > 1
    assert pz.zoom[1] > 1

    # Unzoom with keyboard.
    key_press(qtbot, c, 'Minus')
    assert pz.pan == [0, 0]
    assert pz.zoom == [1, 1]


def test_panzoom_resize(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    c.resize(400, 600)
    assert tuple(pz._canvas_aspect) not in ((0, 0), (1, 1), (1, 0), (0, 1))


def test_panzoom_excluded(qtbot, canvas_pz):
    c = canvas_pz
    visual = MyTestVisual()
    c.add_visual(visual, exclude_origins=(c.panzoom,))
    visual.set_data()

    c.show()
    qtbot.waitForWindowShown(c)

    mouse_drag(qtbot, c, (100, 0), (200, 0))

    # qtbot.stop()
    c.close()
