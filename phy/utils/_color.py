# -*- coding: utf-8 -*-

"""Color routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

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


#------------------------------------------------------------------------------
# Default colormap
#------------------------------------------------------------------------------

# Default color map for the selected clusters.
_COLORMAP = np.array([[8, 146, 252],
                      [255, 2, 2],
                      [240, 253, 2],
                      [228, 31, 228],
                      [2, 217, 2],
                      [255, 147, 2],
                      ])


def _selected_clusters_colors(n_clusters=None):
    if n_clusters is None:
        n_clusters = _COLORMAP.shape[0]
    if n_clusters > _COLORMAP.shape[0]:
        colors = np.tile(_COLORMAP, (1 + n_clusters // _COLORMAP.shape[0], 1))
    else:
        colors = _COLORMAP
    return colors[:n_clusters, ...] / 255.
