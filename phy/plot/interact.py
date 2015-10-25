# -*- coding: utf-8 -*-

"""Common interacts."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import math

import numpy as np
from vispy.gloo import Texture2D

from .base import BaseInteract
from .transform import Scale, Range, Subplot, Clip, NDC
from .utils import _get_texture


#------------------------------------------------------------------------------
# Grid interact
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
                'uniform float u_grid_zoom;\n', '')

    def get_transforms(self):
        # Define the grid transform and clipping.
        m = 1. - .05  # Margin.
        return [Scale(scale='u_grid_zoom'),
                Scale(scale=(m, m)),
                Clip(bounds=[-m, -m, m, m]),
                Subplot(shape=self.shape, index='a_box_index'),
                ]

    def update_program(self, program):
        program['u_grid_zoom'] = self._zoom
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


#------------------------------------------------------------------------------
# Boxed interact
#------------------------------------------------------------------------------

class Boxed(BaseInteract):
    """Boxed interact.

    NOTE: to be used in a boxed, a visual must define `a_box_index`.

    """
    def __init__(self, box_bounds, box_var=None):
        super(Boxed, self).__init__()

        # Name of the variable with the box index.
        self.box_var = box_var or 'a_box_index'

        self.box_bounds = np.atleast_2d(box_bounds)
        assert self.box_bounds.shape[1] == 4
        self.n_boxes = len(self.box_bounds)

    def get_shader_declarations(self):
        return ('#include "utils.glsl"\n\n'
                'attribute float {};\n'.format(self.box_var) +
                'uniform sampler2D u_box_bounds;\n'
                'uniform float n_boxes;', '')

    def get_pre_transforms(self):
        return """
            // Fetch the box bounds for the current box (`box_var`).
            vec4 box_bounds = fetch_texture({},
                                            u_box_bounds,
                                            n_boxes);
            box_bounds = (2 * box_bounds - 1);  // See hack in Python.
            """.format(self.box_var)

    def get_transforms(self):
        return [Range(from_bounds=NDC,
                      to_bounds='box_bounds'),
                ]

    def update_program(self, program):
        # Signal bounds (positions).
        box_bounds = _get_texture(self.box_bounds, NDC, self.n_boxes, [-1, 1])
        program['u_box_bounds'] = Texture2D(box_bounds)
        program['n_boxes'] = self.n_boxes


class Stacked(BaseInteract):
    """Stacked interact.

    NOTE: to be used in a stacked, a visual must define `a_box_index`.

    """

    # # Signal bounds.
    # b = np.zeros((n_signals, 4))
    # b[:, 0] = -1
    # b[:, 1] = np.linspace(-1, 1 - 2. / n_signals, n_signals)
    # b[:, 2] = 1
    # b[:, 3] = np.linspace(-1 + 2. / n_signals, 1., n_signals)
