# -*- coding: utf-8 -*-

"""Test base."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
from pytest import fixture

from ..base import BaseVisual, GLSLInserter, gloo
from ..transform import (subplot_bounds, Translate, Scale, Range,
                         Clip, Subplot, TransformChain)
from . import mouse_click, mouse_drag, mouse_press, key_press, key_release
from phy.gui.qt import QOpenGLWindow

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def vertex_shader_nohook():
    yield """
        attribute vec2 a_position;
        void main() {
            gl_Position = vec4(a_position.xy, 0, 1);
        }
        """


@fixture
def vertex_shader():
    yield """
        attribute vec2 a_position;
        void main() {
            gl_Position = transform(a_position.xy);
            gl_PointSize = 2.0;
        }
        """


@fixture
def fragment_shader():
    yield """
            void main() {
                gl_FragColor = vec4(1, 1, 1, 1);
            }
        """


class MyVisual(BaseVisual):
    def __init__(self):
        super(MyVisual, self).__init__()
        self.set_shader('simple')
        self.set_primitive_type('lines')

    def set_data(self):
        self.n_vertices = 2
        self.program['a_position'] = [[-1, 0], [1, 0]]
        self.program['u_color'] = [1, 1, 1, 1]


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
    tc = TransformChain([Scale(.5)])
    inserter.add_gpu_transforms(tc)
    vs, fs = inserter.insert_into_shaders(vertex_shader, fragment_shader)
    # assert 'temp_pos_tr = temp_pos_tr * 0.5;' in vs
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
    v = MyVisual()
    canvas.add_visual(v, key='key')
    # Should be a no-op when adding the same visual twice.
    canvas.add_visual(v, key='key')
    # Must be called *after* add_visual().
    v.set_data()

    canvas.show()
    qtbot.waitForWindowShown(canvas)

    v.hide()
    canvas.update()
    qtbot.wait(5)
    v.show()
    v.toggle()
    v.toggle()

    assert canvas.get_visual('key') == v
    canvas.remove(v)
    assert canvas.get_visual('key') is None
    canvas.clear()

    # qtbot.stop()


def test_visual_2(qtbot, canvas, vertex_shader, fragment_shader):
    """Test a BaseVisual with multiple CPU and GPU transforms.

    There should be points filling the entire right upper (2, 3) subplot.

    """

    class MyVisual2(BaseVisual):
        def __init__(self):
            super(MyVisual2, self).__init__()
            self.vertex_shader = vertex_shader
            self.fragment_shader = fragment_shader
            self.set_primitive_type('points')
            self.transforms.add(Scale((.1, .1)))
            self.transforms.add(Translate((-1, -1)))
            self.transforms.add(Range(
                (-1, -1, 1, 1), (-1.5, -1.5, 1.5, 1.5)))
            s = 'gl_Position.y += (1 + 1e-8 * u_window_size.x);'
            self.inserter.insert_vert(s, 'after_transforms')
            self.inserter.add_varying('float', 'v_var', 'gl_Position.x')

        def set_data(self):
            self.n_vertices = 1000
            data = np.random.uniform(0, 20, (1000, 2))
            pos = self.transforms.apply(data).astype(np.float32)
            self.program['a_position'] = pos

    bounds = subplot_bounds(shape=(2, 3), index=(1, 2))
    canvas.gpu_transforms.add([Subplot((2, 3), (1, 2)), Clip(bounds)])

    # We attach the visual to the canvas. By default, a BaseLayout is used.
    v = MyVisual2()
    canvas.add_visual(v)
    v.set_data()

    v = MyVisual2()
    canvas.add_visual(v)
    v.set_data()

    canvas.show()
    qtbot.waitForWindowShown(canvas)
    # qtbot.stop()


def test_canvas_lazy(qtbot, canvas):
    v = MyVisual()
    canvas.add_visual(v)
    canvas.set_lazy(True)
    v.set_data()
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    assert len(list(canvas.iter_update_queue())) == 2


def test_visual_benchmark(qtbot, vertex_shader_nohook, fragment_shader):
    try:
        from memory_profiler import memory_usage
    except ImportError:  # pragma: no cover
        logger.warning("Skip test depending on unavailable memory_profiler module.")
        return

    class TestCanvas(QOpenGLWindow):
        def paintGL(self):
            gloo.clear()
            program.draw('points')

    program = gloo.Program(vertex_shader_nohook, fragment_shader)

    canvas = TestCanvas()
    canvas.show()
    qtbot.waitForWindowShown(canvas)

    def f():
        for _ in range(100):
            program['a_position'] = (-1 + 2 * np.random.rand(100_000, 2)).astype(np.float32)
            canvas.update()
            qtbot.wait(1)

    mem = memory_usage(f)
    usage = max(mem) - min(mem)
    print(usage)

    # NOTE: this test is failing currently because of a memory leak in the the gloo module.
    # Recreating a buffer at every cluster selection causes a memory leak, once should ideally
    # use a single large buffer and reuse that, even if the buffer's content is actually smaller.
    # assert usage < 10

    canvas.close()
