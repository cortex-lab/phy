# -*- coding: utf-8 -*-

"""Test base."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..base import BaseVisual, BaseInteract
from ..transform import Scale


#------------------------------------------------------------------------------
# Test base
#------------------------------------------------------------------------------

def test_visual_shader_name(qtbot, canvas):
    """Test a BaseVisual with a shader name."""
    class TestVisual(BaseVisual):
        shader_name = 'box'
        gl_primitive_type = 'lines'

        def set_data(self):
            self.data['a_position'] = [[-1, 0, 0], [1, 0, 0]]
            self.data['n_rows'] = 1

    v = TestVisual()
    v.set_data()
    # We need to build the program explicitly when there is no interact.
    v.attach(canvas)
    v.build_program()

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

        def set_data(self):
            self.data['a_position'] = [[-1, 0], [1, 0]]

    v = TestVisual()
    v.set_data()
    # We need to build the program explicitly when there is no interact.
    v.attach(canvas)
    v.build_program()

    canvas.show()
    v.hide()
    v.show()
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()

    # Simulate a mouse move.
    canvas.events.mouse_move(delta=(1., 0.))
    canvas.events.key_press(text='a')

    v.update()


def test_base_interact(qtbot, canvas):
    """Test a BaseVisual with a CPU transform and a blank interact."""
    class TestVisual(BaseVisual):
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
            super(TestVisual, self).__init__()
            self.set_data()

        def set_data(self):
            self.data['a_position'] = [[-1, 0], [1, 0]]
            self.transforms = [Scale(scale=(.5, 1))]

    # We attach the visual to the canvas. By default, a BaseInteract is used.
    v = TestVisual()
    v.attach(canvas)

    # Base interact (no transform).
    interact = BaseInteract()
    interact.attach(canvas)

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()
