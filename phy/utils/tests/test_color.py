# -*- coding: utf-8 -*-

"""Test colors."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from .._color import (_random_color, _is_bright, _random_bright_color,
                      _colormap, _spike_colors, ColorSelector,
                      )
from ..testing import show_colored_canvas


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_random_color():
    color = _random_color()
    show_colored_canvas(color)

    for _ in range(10):
        assert _is_bright(_random_bright_color())


def test_colormap():
    assert len(_colormap(0)) == 3
    assert len(_colormap(1000)) == 3

    assert _spike_colors([0, 1, 10, 1000]).shape == (4, 4)
    assert _spike_colors([0, 1, 10, 1000],
                         alpha=1.).shape == (4, 4)
    assert _spike_colors([0, 1, 10, 1000],
                         masks=np.linspace(0., 1., 4)).shape == (4, 4)
    assert _spike_colors(masks=np.linspace(0., 1., 4)).shape == (4, 4)


def test_color_selector():
    sel = ColorSelector()
    assert len(sel.get(0)) == 4
    assert len(sel.get(0, [1, 0])) == 4
    assert sel.get(0, cluster_group='noise') == (.5,) * 4
