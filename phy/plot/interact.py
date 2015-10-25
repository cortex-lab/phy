# -*- coding: utf-8 -*-

"""Common interacts."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import math

from .base import BaseInteract
from .transform import Scale, Subplot, Clip


#------------------------------------------------------------------------------
# Grid class
#------------------------------------------------------------------------------

class Grid(BaseInteract):
    """Grid interact.

    NOTE: to be used in a grid, a visual must define `a_box_index`.

    """

    def __init__(self, shape, box_var=None):
        super(Grid, self).__init__()
        self._zoom = 1.

        # Name of the variable with the box index.
        self.box_var = box_var or 'a_box_index'

        self.shape = shape
        assert len(shape) == 2
        assert shape[0] >= 1
        assert shape[1] >= 1

    def get_shader_declarations(self):
        return ('attribute vec2 a_box_index;\n'
                'uniform float u_zoom;\n', '')

    def get_transforms(self):
        # Define the grid transform and clipping.
        m = 1. - .05  # Margin.
        return [Scale(scale='u_zoom'),
                Scale(scale=(m, m)),
                Clip(bounds=[-m, -m, m, m]),
                Subplot(shape=self.shape, index='a_box_index'),
                ]

    def update_program(self, program):
        program['u_zoom'] = self._zoom
        # Only set the default box index if necessary.
        try:
            program['a_box_index']
        except KeyError:
            program['a_box_index'] = (0, 0)

    @property
    def zoom(self):
        """Zoom level."""
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        """Zoom level."""
        self._zoom = value
        self.update()

    def on_key_press(self, event):
        """Pan and zoom with the keyboard."""
        super(Grid, self).on_key_press(event)
        if event.modifiers:
            return
        key = event.key

        # Zoom.
        if key in ('-', '+'):
            k = .05 if key == '+' else -.05
            self.zoom *= math.exp(1.5 * k)
            self.update()

        # Reset with 'R'.
        if key == 'R':
            self.zoom = 1.
            self.update()
