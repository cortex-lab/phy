# -*- coding: utf-8 -*-

"""Tests of misc type utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from .._types import Bunch, _is_integer


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_bunch():
    obj = Bunch()
    obj['a'] = 1
    assert obj.a == 1
    obj.b = 2
    assert obj['b'] == 2


def test_integer():
    assert _is_integer(3)
    assert _is_integer(np.arange(1)[0])
    assert not _is_integer(3.)
