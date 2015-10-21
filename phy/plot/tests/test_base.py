# -*- coding: utf-8 -*-

"""Test base."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..base import BaseVisual


#------------------------------------------------------------------------------
# Test base
#------------------------------------------------------------------------------

def test_base_visual(qtbot, canvas):

    class TestVisual(BaseVisual):
        shader_name = 'test'
        gl_primitive_type = 'lines'

        def set_data(self):
            self.build_program()
            self.program['a_position'] = [[-1, 0], [1, 0]]
            self.show()

    v = TestVisual()
    v.set_data()
    v.attach(canvas)

    canvas.show()
    v.hide()
    v.show()
    qtbot.waitForWindowShown(canvas.native)
    # qtbot.stop()

    # Simulate a mouse move.
    canvas.events.mouse_move(delta=(1., 0.))

    v.update()
