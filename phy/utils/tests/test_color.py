# -*- coding: utf-8 -*-

"""Test colors."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

from .._color import (_random_color, _is_bright, _random_bright_color,
                      _selected_clusters_colors,
                      )
from ..testing import show_colored_canvas


# Skip these tests in "make test-quick".
pytestmark = mark.long


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_random_color():
    color = _random_color()
    show_colored_canvas(color)

    for _ in range(10):
        assert _is_bright(_random_bright_color())


def test_selected_clusters_colors():
    assert _selected_clusters_colors().ndim == 2
    assert len(_selected_clusters_colors(3)) == 3
    assert len(_selected_clusters_colors(10)) == 10
