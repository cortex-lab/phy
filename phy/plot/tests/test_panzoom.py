# -*- coding: utf-8 -*-

"""Test panzoom."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_allclose as ac
from pytest import yield_fixture
from vispy.app import MouseEvent
from vispy.util import keys

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
        self.program['a_position'] = [[-1, 0], [1, 0]]
        self.program['u_color'] = [1, 1, 1, 1]


@yield_fixture
def panzoom(qtbot, canvas_pz):
    c = canvas_pz
    visual = MyTestVisual()
    c.add_visual(visual)
    visual.set_data()

    c.show()
    qtbot.waitForWindowShown(c.native)

    yield c.panzoom


#------------------------------------------------------------------------------
# Test panzoom
#------------------------------------------------------------------------------

def test_panzoom_basic_attrs():
    pz = PanZoom()

    # Aspect.
    assert pz.aspect == 1.
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
    pz = PanZoom(constrain_bounds=(-1, -1, 1, 1))

    # Aspect.
    assert pz.aspect == 1.
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
    pos = pz.get_mouse_pos((.01, -.01))
    ac(pos, (.5, .25), atol=1e-3)


#------------------------------------------------------------------------------
# Test panzoom on canvas
#------------------------------------------------------------------------------

def test_panzoom_pan_mouse(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Pan with mouse.
    press = MouseEvent(type='mouse_press', pos=(0, 0))
    c.events.mouse_move(pos=(10., 0.), button=1,
                        last_event=press, press_event=press)
    assert pz.pan[0] > 0
    assert pz.pan[1] == 0
    pz.pan = (0, 0)

    # Panning with a modifier should not pan.
    press = MouseEvent(type='mouse_press', pos=(0, 0))
    c.events.mouse_move(pos=(10., 0.), button=1,
                        last_event=press, press_event=press,
                        modifiers=(keys.CONTROL,))
    assert pz.pan == [0, 0]

    # qtbot.stop()


def test_panzoom_touch(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Pan with mouse.
    c.events.touch(type='pinch', pos=(0, 0), scale=1, last_scale=1)
    c.events.touch(type='pinch', pos=(0, 0), scale=2, last_scale=1)
    assert pz.zoom[0] >= 2
    c.events.touch(type='end')

    c.events.touch(type='touch', pos=(0.1, 0), last_pos=(0, 0))
    assert pz.pan[0] >= 1


def test_panzoom_pan_keyboard(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Pan with keyboard.
    c.events.key_press(key=keys.UP)
    assert pz.pan[0] == 0
    assert pz.pan[1] < 0

    # All panning movements with keys.
    c.events.key_press(key=keys.LEFT)
    c.events.key_press(key=keys.DOWN)
    c.events.key_press(key=keys.RIGHT)
    assert pz.pan == [0, 0]

    # Reset with R.
    c.events.key_press(key=keys.RIGHT)
    c.events.key_press(key=keys.Key('r'))
    assert pz.pan == [0, 0]

    # Using modifiers should not pan.
    c.events.key_press(key=keys.UP, modifiers=(keys.CONTROL,))
    assert pz.pan == [0, 0]

    # Disable keyboard pan.
    pz.enable_keyboard_pan = False
    c.events.key_press(key=keys.UP, modifiers=(keys.CONTROL,))
    assert pz.pan == [0, 0]


def test_panzoom_zoom_mouse(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Zoom with mouse.
    press = MouseEvent(type='mouse_press', pos=(50., 50.))
    c.events.mouse_move(pos=(0., 0.), button=2,
                        last_event=press, press_event=press)
    assert pz.pan[0] < 0
    assert pz.pan[1] < 0
    assert pz.zoom[0] < 1
    assert pz.zoom[1] > 1
    pz.reset()

    # Zoom with mouse.
    size = np.asarray(c.size)
    c.events.mouse_wheel(pos=size / 2., delta=(0., 1.))
    assert pz.pan == [0, 0]
    assert pz.zoom[0] > 1
    assert pz.zoom[1] > 1
    pz.reset()

    # Using modifiers with the wheel should not zoom.
    c.events.mouse_wheel(pos=(0., 0.), delta=(0., 1.),
                         modifiers=(keys.CONTROL,))
    assert pz.pan == [0, 0]
    assert pz.zoom == [1, 1]
    pz.reset()


def test_panzoom_zoom_keyboard(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Zoom with keyboard.
    c.events.key_press(key=keys.Key('+'))
    assert pz.pan == [0, 0]
    assert pz.zoom[0] > 1
    assert pz.zoom[1] > 1

    # Unzoom with keyboard.
    c.events.key_press(key=keys.Key('-'))
    assert pz.pan == [0, 0]
    assert pz.zoom == [1, 1]


def test_panzoom_resize(qtbot, canvas_pz, panzoom):
    c = canvas_pz
    pz = panzoom

    # Increase coverage with different aspect ratio.
    c.native.resize(400, 600)
    # qtbot.stop()
    # c.events.resize(size=(100, 1000))
    assert list(pz._canvas_aspect) == [1., 2. / 3]
