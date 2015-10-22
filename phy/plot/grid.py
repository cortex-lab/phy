# -*- coding: utf-8 -*-

"""Grid interact for subplots."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from .base import BaseInteract
from .transform import Scale, Subplot, Clip, pixels_to_ndc


#------------------------------------------------------------------------------
# Grid class
#------------------------------------------------------------------------------

class Grid(BaseInteract):
    """Grid interact."""

    def __init__(self, shape, box_var=None):
        """
        """
        super(Grid, self).__init__()
        self.box_var = box_var or 'a_box'
        self.shape = shape
        assert len(shape) == 2
        assert shape[0] >= 1
        assert shape[1] >= 1

        # Define the grid transform and clipping.
        m = 1. - .05
        self.transforms = [Scale(scale=(m, m)),
                           Clip(bounds=[-m, -m, m, m]),
                           Subplot(shape=shape, index='a_box'),
                           ]
        self.vertex_decl = 'attribute vec2 a_box;\n'
