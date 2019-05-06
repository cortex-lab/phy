# -*- coding: utf-8 -*-

"""Test base."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from pytest import yield_fixture

from phy.utilslib import emit
from ..base import BaseVisual, BaseLayout, GLSLInserter
from ..transform import (subplot_bounds, Translate, Scale, Range,
                         Clip, Subplot, TransformChain)
from . import mouse_click, mouse_drag, mouse_press, key_press, key_release


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def vertex_shader_nohook():
    yield """
        attribute vec2 a_position;
        void main() {
            gl_Position = vec4(a_position.xy, 0, 1);
        }
        """


@yield_fixture
def vertex_shader():
    yield """
        attribute vec2 a_position;
        void main() {
            gl_Position = transform(a_position.xy);
            gl_PointSize = 2.0;
        }
        """


@yield_fixture
def fragment_shader():
    yield """
            void main() {
                gl_FragColor = vec4(1, 1, 1, 1);
            }
        """


#------------------------------------------------------------------------------
# Test base
#------------------------------------------------------------------------------

def test_glsl_inserter_nohook(vertex_shader_nohook, fragment_shader):
    vertex_shader = vertex_shader_nohook
    inserter = GLSLInserter()
    inserter.insert_vert('uniform float boo;', 'header')
    inserter.insert_frag('// In fragment shader.', 'before_transforms')
    vs, fs = inserter.insert_into_shaders(vertex_shader, fragment_shader)
    assert vs == vertex_shader
    assert fs == fragment_shader


def test_glsl_inserter_hook(vertex_shader, fragment_shader):
    inserter = GLSLInserter()
    inserter.insert_vert('uniform float boo;', 'header')
    inserter.insert_frag('// In fragment shader.', 'before_transforms')
    tc = TransformChain()
    tc.add_on_gpu([Scale(.5)])
    inserter.add_transform_chain(tc)
    vs, fs = inserter.insert_into_shaders(vertex_shader, fragment_shader)
    assert 'temp_pos_tr = temp_pos_tr * 0.5;' in vs
    assert 'uniform float boo;' in vs
    assert '// In fragment shader.' in fs


def test_mock_events(qtbot, canvas):
    c = canvas
    pos = p0 = (50, 50)
    p1 = (100, 100)
    key = 'A'
    mouse_click(qtbot, c, pos, button='left', modifiers=())
    mouse_press(qtbot, c, pos, button='left', modifiers=())
    mouse_drag(qtbot, c, p0, p1, button='left', modifiers=())
    key_press(qtbot, c, key, modifiers=())
    key_release(qtbot, c, key, modifiers=())


def test_next_paint(qtbot, canvas):
    @canvas.on_next_paint
    def next():
        pass
    canvas.show()
    qtbot.waitForWindowShown(canvas)


def test_visual_1(qtbot, canvas):
    class TestVisual(BaseVisual):
        def __init__(self):
            super(TestVisual, self).__init__()
            self.set_shader('simple')
            self.set_primitive_type('lines')

        def set_data(self):
            self.n_vertices = 2
            self.program['a_position'] = [[-1, 0], [1, 0]]
            self.program['u_color'] = [1, 1, 1, 1]

    v = TestVisual()
    canvas.add_visual(v, key='key')
    # Must be called *after* add_visual().
    v.set_data()

    canvas.show()
    qtbot.waitForWindowShown(canvas)

    assert canvas.get_visual('key') == v
    canvas.remove(v)
    assert canvas.get_visual('key') is None
    canvas.clear()

    # qtbot.stop()


def test_visual_2(qtbot, canvas, vertex_shader, fragment_shader):
    """Test a BaseVisual with multiple CPU and GPU transforms.

    There should be points filling the entire right upper (2, 3) subplot.

    """

    class TestVisual(BaseVisual):
        def __init__(self):
            super(TestVisual, self).__init__()
            self.vertex_shader = vertex_shader
            self.fragment_shader = fragment_shader
            self.set_primitive_type('points')
            self.transforms.add_on_cpu(Scale((.1, .1)))
            self.transforms.add_on_cpu(Translate((-1, -1)))
            self.transforms.add_on_cpu(Range((-1, -1, 1, 1),
                                             (-1.5, -1.5, 1.5, 1.5),
                                             ))
            s = 'gl_Position.y += (1 + 1e-8 * u_window_size.x);'
            self.inserter.insert_vert(s, 'after_transforms')

        def set_data(self):
            self.n_vertices = 1000
            data = np.random.uniform(0, 20, (1000, 2))
            pos = self.transforms.apply(data).astype(np.float32)
            self.program['a_position'] = pos

    bounds = subplot_bounds(shape=(2, 3), index=(1, 2))
    canvas.transforms.add_on_gpu([Subplot((2, 3), (1, 2)),
                                  Clip(bounds),
                                  ])

    # We attach the visual to the canvas. By default, a BaseLayout is used.
    v = TestVisual()
    canvas.add_visual(v)
    v.set_data()

    v = TestVisual()
    canvas.add_visual(v)
    v.set_data()

    canvas.show()
    qtbot.waitForWindowShown(canvas)
    # qtbot.stop()


def test_layout_1(qtbot, canvas):
    layout = BaseLayout()
    layout.attach(canvas)
    #layout.update()

    class TestVisual(BaseVisual):
        def __init__(self):
            super(TestVisual, self).__init__()
            self.set_shader('simple')
            self.set_primitive_type('lines')

        def set_data(self):
            self.n_vertices = 2
            self.program['a_position'] = [[-1, 0], [1, 0]]
            self.program['u_color'] = [1, 1, 1, 1]
            emit('visual_set_data', self)

    v = TestVisual()
    canvas.add_visual(v)
    v.set_data()

    canvas.show()
    qtbot.waitForWindowShown(canvas)
    layout.update()
