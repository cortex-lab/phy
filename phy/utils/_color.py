# -*- coding: utf-8 -*-

"""Color routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from random import uniform
from colorsys import hsv_to_rgb


#------------------------------------------------------------------------------
# Colors
#------------------------------------------------------------------------------

def _random_color():
    """Generate a random RGB color."""
    h, s, v = uniform(0., 1.), uniform(.5, 1.), uniform(.5, 1.)
    r, g, b = hsv_to_rgb(h, s, v)
    return r, g, b


def _is_bright(rgb):
    """Return whether a RGB color is bright or not."""
    r, g, b = rgb
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray >= .5


def _random_bright_color():
    """Generate a random bright color."""
    rgb = _random_color()
    while not _is_bright(rgb):
        rgb = _random_color()
    return rgb
