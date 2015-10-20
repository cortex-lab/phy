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
        _shader_name = 'box'
        _gl_primitive_type = 'lines'

        def set_data(self):
            self.program['a_position'] = [[-1, 0, 0], [1, 0, 0]]
            self.program['n_rows'] = 1
            self.show()

    v = TestVisual()
    v.set_data()
    canvas.add_visual(v)

    canvas.show()
    v.hide()
    v.show()
    qtbot.waitForWindowShown(canvas.native)

    # Simulate a mouse move.
    canvas.events.mouse_move(delta=(1., 0.))

    v.update()
