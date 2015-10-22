# -*- coding: utf-8 -*-

"""Test panzoom."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from vispy.app import MouseEvent
from vispy.util import keys
from pytest import yield_fixture

from ..base import BaseVisual
from ..panzoom import PanZoom
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
def panzoom(qtbot, canvas, visual):
    visual.attach(canvas, 'PanZoom')
    visual.show()

    pz = PanZoom()
    pz.attach(canvas)

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)

    yield pz


#------------------------------------------------------------------------------
# Test panzoom
#------------------------------------------------------------------------------

def test_panzoom_basic_attrs():
    pz = PanZoom()

    assert not pz.is_attached()

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

    assert list(pz.iter_attached_visuals()) == []


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


def test_panzoom_pan_mouse(qtbot, canvas, panzoom):
    pz = panzoom

    # Pan with mouse.
    press = MouseEvent(type='mouse_press', pos=(0, 0))
    canvas.events.mouse_move(pos=(10., 0.), button=1,
                             last_event=press, press_event=press)
    assert pz.pan[0] > 0
    assert pz.pan[1] == 0
    pz.pan = (0, 0)

    # Panning with a modifier should not pan.
    press = MouseEvent(type='mouse_press', pos=(0, 0))
    canvas.events.mouse_move(pos=(10., 0.), button=1,
                             last_event=press, press_event=press,
                             modifiers=(keys.CONTROL,))
    assert pz.pan == [0, 0]


def test_panzoom_pan_keyboard(qtbot, canvas, panzoom):
    pz = panzoom

    # Pan with keyboard.
    canvas.events.key_press(key=keys.UP)
    assert pz.pan[0] == 0
    assert pz.pan[1] < 0

    # All panning movements with keys.
    canvas.events.key_press(key=keys.LEFT)
    canvas.events.key_press(key=keys.DOWN)
    canvas.events.key_press(key=keys.RIGHT)
    assert pz.pan == [0, 0]

    # Reset with R.
    canvas.events.key_press(key=keys.RIGHT)
    canvas.events.key_press(key=keys.Key('r'))
    assert pz.pan == [0, 0]

    # Using modifiers should not pan.
    canvas.events.key_press(key=keys.UP, modifiers=(keys.CONTROL,))
    assert pz.pan == [0, 0]


def test_panzoom_zoom_mouse(qtbot, canvas, panzoom):
    pz = panzoom

    # Zoom with mouse.
    press = MouseEvent(type='mouse_press', pos=(50., 50.))
    canvas.events.mouse_move(pos=(0., 0.), button=2,
                             last_event=press, press_event=press)
    assert pz.pan[0] < 0
    assert pz.pan[1] < 0
    assert pz.zoom[0] < 1
    assert pz.zoom[1] > 1
    pz.reset()

    # Zoom with mouse.
    size = np.asarray(canvas.size)
    canvas.events.mouse_wheel(pos=size / 2., delta=(0., 1.))
    assert pz.pan == [0, 0]
    assert pz.zoom[0] > 1
    assert pz.zoom[1] > 1
    pz.reset()

    # Using modifiers with the wheel should not zoom.
    canvas.events.mouse_wheel(pos=(0., 0.), delta=(0., 1.),
                              modifiers=(keys.CONTROL,))
    assert pz.pan == [0, 0]
    assert pz.zoom == [1, 1]
    pz.reset()


def test_panzoom_zoom_keyboard(qtbot, canvas, panzoom):
    pz = panzoom

    # Zoom with keyboard.
    canvas.events.key_press(key=keys.Key('+'))
    assert pz.pan == [0, 0]
    assert pz.zoom[0] > 1
    assert pz.zoom[1] > 1

    # Unzoom with keyboard.
    canvas.events.key_press(key=keys.Key('-'))
    assert pz.pan == [0, 0]
    assert pz.zoom == [1, 1]


def test_panzoom_resize(qtbot, canvas, panzoom):
    # Increase coverage with different aspect ratio.
    canvas.native.resize(400, 600)
    # canvas.events.resize(size=(100, 1000))
    assert list(panzoom._canvas_aspect) == [1., 2. / 3]
