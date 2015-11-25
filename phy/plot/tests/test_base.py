# -*- coding: utf-8 -*-

"""Test base."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent

import numpy as np

from ..base import BaseVisual, BaseInteract, insert_glsl
from ..transform import (subplot_bounds, Translate, Scale, Range,
                         Clip, Subplot, TransformChain)


#------------------------------------------------------------------------------
# Test base
#------------------------------------------------------------------------------

def test_visual_shader_name(qtbot, canvas):
    """Test a BaseVisual with a shader name."""
    class TestVisual(BaseVisual):

        def __init__(self):
            super(TestVisual, self).__init__()
            self.set_shader('simple')
            self.set_primitive_type('lines')

        def set_data(self):
            self.program['a_position'] = [[-1, 0], [1, 0]]
            self.program['u_color'] = [1, 1, 1, 1]

    v = TestVisual()
    # We need to build the program explicitly when there is no interact.
    v.attach(canvas)
    # Must be called *after* attach().
    v.set_data()

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()


def test_base_visual(qtbot, canvas):
    """Test a BaseVisual with custom shaders."""

    class TestVisual(BaseVisual):

        def __init__(self):
            super(TestVisual, self).__init__()
            self.vertex_shader = """
                attribute vec2 a_position;
                void main() {
                    gl_Position = vec4(a_position.xy, 0, 1);
                }
                """
            self.fragment_shader = """
                void main() {
                    gl_FragColor = vec4(1, 1, 1, 1);
                }
            """
            self.set_primitive_type('lines')

        def set_data(self):
            self.program['a_position'] = [[-1, 0], [1, 0]]

    v = TestVisual()
    # We need to build the program explicitly when there is no interact.
    v.attach(canvas)
    v.set_data()

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()

    # Simulate a mouse move.
    canvas.events.mouse_move(pos=(0., 0.))
    canvas.events.key_press(text='a')

    v.update()


def test_base_interact():
    interact = BaseInteract()
    assert interact.get_shader_declarations() == ('', '')
    assert interact.get_pre_transforms() == ''
    assert interact.get_transforms() == []
    interact.update_program(None)


def test_no_interact(qtbot, canvas):
    """Test a BaseVisual with a CPU transform and no interact."""
    class TestVisual(BaseVisual):
        def __init__(self):
            super(TestVisual, self).__init__()
            self.set_shader('simple')
            self.set_primitive_type('lines')
            self.transforms.add_on_cpu(Scale(scale=(.5, 1)))

        def set_data(self):
            self.program['a_position'] = [[-1, 0], [1, 0]]
            self.program['u_color'] = [1, 1, 1, 1]

    # We attach the visual to the canvas. By default, a BaseInteract is used.
    v = TestVisual()
    v.attach(canvas)
    v.set_data()

    canvas.show()
    assert not canvas.interacts
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()


def test_interact(qtbot, canvas):
    """Test a BaseVisual with multiple CPU and GPU transforms and a
    non-blank interact.

    There should be points filling the entire lower (2, 3) subplot.

    """

    class TestVisual(BaseVisual):
        def __init__(self):
            super(TestVisual, self).__init__()
            self.vertex_shader = """
                attribute vec2 a_position;
                void main() {
                    gl_Position = transform(a_position);
                    gl_PointSize = 2.0;
                }
            """
            self.fragment_shader = """
                void main() {
                    gl_FragColor = vec4(1, 1, 1, 1);
                }
            """
            self.set_primitive_type('points')
            self.transforms.add_on_cpu(Scale(scale=(.1, .1)))
            self.transforms.add_on_cpu(Translate(translate=(-1, -1)))
            self.transforms.add_on_cpu(Range(from_bounds=(-1, -1, 1, 1),
                                             to_bounds=(-1.5, -1.5, 1.5, 1.5),
                                             ))
            self.insert_vert("""gl_Position.y += 1;""", 'after_transforms')

        def set_data(self):
            data = np.random.uniform(0, 20, (1000, 2)).astype(np.float32)
            self.program['a_position'] = self.transforms.apply(data)

    class TestInteract(BaseInteract):
        def get_transforms(self):
            bounds = subplot_bounds(shape=(2, 3), index=(1, 2))
            return [Subplot(shape=(2, 3), index=(1, 2)),
                    Clip(bounds=bounds),
                    ]

    TestInteract().attach(canvas)

    # We attach the visual to the canvas. By default, a BaseInteract is used.
    v = TestVisual()
    v.attach(canvas)
    v.set_data()

    canvas.show()
    assert len(canvas.interacts) == 1
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()


def test_transform_chain_complete():
    t = TransformChain([Scale(scale=.5),
                        Scale(scale=2.)])
    t.add_on_cpu([Range(from_bounds=[-3, -3, 1, 1])])
    t.add_on_gpu([Clip(),
                  Subplot(shape='u_shape', index='a_box_index'),
                  ])

    vs = dedent("""
    attribute vec2 a_position;
    void main() {
        gl_Position = transform(a_position);
    }
    """).strip()

    fs = dedent("""
    void main() {
        gl_FragColor = vec4(1., 1., 1., 1.);
    }
    """).strip()
    vs, fs = insert_glsl(t, vs, fs)
    assert 'a_box_index' in vs
    assert 'v_' in vs
    assert 'v_' in fs
    assert 'discard' in fs

    # Increase coverage.
    insert_glsl(t, vs.replace('transform', ''), fs)
