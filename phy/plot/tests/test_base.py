# -*- coding: utf-8 -*-

"""Test base."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..base import BaseVisual, BaseInteract
from ..transform import (subplot_bounds, Translate, Scale, Range,
                         Clip, Subplot, GPU)


#------------------------------------------------------------------------------
# Test base
#------------------------------------------------------------------------------

def test_visual_shader_name(qtbot, canvas):
    """Test a BaseVisual with a shader name."""
    class TestVisual(BaseVisual):
        shader_name = 'simple'
        gl_primitive_type = 'lines'

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
        vertex = """
            attribute vec2 a_position;
            void main() {
                gl_Position = vec4(a_position.xy, 0, 1);
            }
            """
        fragment = """
            void main() {
                gl_FragColor = vec4(1, 1, 1, 1);
            }
        """
        gl_primitive_type = 'lines'

        def get_shaders(self):
            return self.vertex, self.fragment

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
        shader_name = 'simple'
        gl_primitive_type = 'lines'

        def get_transforms(self):
            return [Scale(scale=(.5, 1))]

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
        vertex = """
            attribute vec2 a_position;
            void main() {
                gl_Position = transform(a_position);
                gl_PointSize = 2.0;
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

        def get_transforms(self):
            return [Scale(scale=(.1, .1)),
                    Translate(translate=(-1, -1)),
                    GPU(),
                    Range(from_bounds=(-1, -1, 1, 1),
                          to_bounds=(-1.5, -1.5, 1.5, 1.5),
                          ),
                    ]

        def get_post_transforms(self):
            return """
                gl_Position.y += 1;
            """

        def set_data(self):
            data = np.random.uniform(0, 20, (1000, 2)).astype(np.float32)
            self.program['a_position'] = self.apply_cpu_transforms(data)

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
