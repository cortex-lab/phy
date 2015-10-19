# -*- coding: utf-8 -*-

"""Test base."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from vispy import gloo

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

        def on_mouse_move(self, e):
            y = 1 - 2 * e.pos[1] / float(self.size[1])
            self.program['a_position'] = [[-1, y, 0], [1, y, 0]]
            self.update()

    def on_draw(e):
        gloo.clear()

    canvas.events['draw'].connect(on_draw, position='last')

    v = TestVisual()
    v.set_data()
    canvas.add_visual(v)

    canvas.show()
    # qtbot.stop()
