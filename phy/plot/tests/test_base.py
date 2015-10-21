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

def test_base_visual(qtbot, canvas):

    class TestVisual(BaseVisual):
        shader_name = 'test'
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

    v.update()


def test_base_interact(qtbot, canvas):

    class TestVisual(BaseVisual):
        shader_name = 'test'
        gl_primitive_type = 'lines'

        def __init__(self):
            super(TestVisual, self).__init__()
            self.set_data()

        def set_data(self):
            self.data['a_position'] = [[-1, 0], [1, 0]]
            self.transforms = [Scale((.5, 1))]

    v = TestVisual()
    v.attach(canvas)

    interact = BaseInteract()
    interact.attach(canvas)

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()

    v.update()
