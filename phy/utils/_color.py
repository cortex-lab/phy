# -*- coding: utf-8 -*-

"""Color routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.random import uniform
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


#------------------------------------------------------------------------------
# Random colors
#------------------------------------------------------------------------------

def _random_color(h_range=(0., 1.),
                  s_range=(.5, 1.),
                  v_range=(.5, 1.),
                  ):
    """Generate a random RGB color."""
    h, s, v = uniform(*h_range), uniform(*s_range), uniform(*v_range)
    r, g, b = hsv_to_rgb(np.array([h, s, v])).flat
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
# Colormap
#------------------------------------------------------------------------------

# Default color map for the selected clusters.
_COLORMAP = np.array([[8, 146, 252],
                      [255, 2, 2],
                      [240, 253, 2],
                      [228, 31, 228],
                      [2, 217, 2],
                      [255, 147, 2],

                      [212, 150, 70],
                      [205, 131, 201],
                      [201, 172, 36],
                      [150, 179, 62],
                      [95, 188, 122],
                      [129, 173, 190],
                      [231, 107, 119],
                      ])


def _apply_color_masks(color, masks=None, alpha=None):
    alpha = alpha or .5
    hsv = rgb_to_hsv(color[:, :3])
    # Change the saturation and value as a function of the mask.
    if masks is not None:
        hsv[:, 1] *= masks
        hsv[:, 2] *= .5 * (1. + masks)
    color = hsv_to_rgb(hsv)
    n = color.shape[0]
    color = np.c_[color, alpha * np.ones((n, 1))]
    return color


def _colormap(i):
    n = len(_COLORMAP)
    return _COLORMAP[i % n] / 255.


def _spike_colors(spike_clusters=None, masks=None, alpha=None):
    n = len(_COLORMAP)
    if spike_clusters is not None:
        c = _COLORMAP[np.mod(spike_clusters, n), :] / 255.
    else:
        c = np.ones((masks.shape[0], 3))
    c = _apply_color_masks(c, masks=masks, alpha=alpha)
    return c


class ColorSelector(object):
    """Return the color of a cluster.

    If the cluster belongs to the selection, returns the colormap color.

    Otherwise, return a random color and remember this color.

    """
    def __init__(self):
        self._colors = {}

    def get(self, clu, cluster_ids=None, cluster_group=None, alpha=None):
        alpha = alpha or .5
        if cluster_group in ('noise', 'mua'):
            color = (.5,) * 4
        elif cluster_ids and clu in cluster_ids:
            i = cluster_ids.index(clu)
            color = _colormap(i)
            color = tuple(color) + (alpha,)
        else:
            if clu in self._colors:
                return self._colors[clu]
            color = _random_color(v_range=(.3, .6))
            color = tuple(color) + (alpha,)
            self._colors[clu] = color
        assert len(color) == 4
        return color
