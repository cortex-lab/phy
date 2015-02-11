# -*- coding: utf-8 -*-

"""Test colors."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from .._color import _random_color, _is_bright, _random_bright_color
from ..testing import show_colored_canvas


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_random_color():
    color = _random_color()
    show_colored_canvas(color)

    assert _is_bright(_random_bright_color())
