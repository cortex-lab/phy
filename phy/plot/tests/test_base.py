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
            self._data['a_position'] = [[-1, 0, 0], [1, 0, 0]]
            self._data['n_rows'] = 1
            self._to_upload = ['a_position', 'n_rows']

        def is_empty(self):
            return False

    def on_draw(e):
        gloo.clear()

    canvas.events['draw'].connect(on_draw, position='last')

    v = TestVisual()
    v.set_data()

    v.attach(canvas)
    canvas.show()

    # qtbot.stop()
