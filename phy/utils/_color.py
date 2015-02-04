# -*- coding: utf-8 -*-

"""Color routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from random import uniform
from colorsys import rgb_to_hsv, hsv_to_rgb

import numpy as np


#------------------------------------------------------------------------------
# Colors
#------------------------------------------------------------------------------

def _random_color():
    """Generate a random RGB color."""
    h, s, v = uniform(0., 1.), uniform(.5, 1.), uniform(.5, 1.)
    r, g, b = hsv_to_rgb(h, s, v)
    return r, g, b
