# -*- coding: utf-8 -*-

"""Test panzoom."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

# import numpy as np
from pytest import yield_fixture

from ..base import BaseVisual
from ..panzoom import PanZoom


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
        self.set_data()

    def set_data(self):
        self.data['a_position'] = [[-1, 0], [1, 0]]


@yield_fixture
def visual():
    yield MyTestVisual()


#------------------------------------------------------------------------------
# Test panzoom
#------------------------------------------------------------------------------

def test_pz_basic_attrs():
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


def test_pz_basic_pan_zoom():
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


def test_pz_attached(qtbot, canvas, visual):

    visual.attach(canvas)

    pz = PanZoom()
    pz.attach(canvas)

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()
