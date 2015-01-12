# -*- coding: utf-8 -*-

"""Tests of utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy import array_equal as ae
from pytest import raises

from ..utils import _range_from_slice


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

class _SliceTest(object):
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _range_from_slice(item, **self._kwargs)


def test_range_from_slice():
    """Test '_range_from_slice'."""
    with raises(ValueError):
        _SliceTest()[:]
    with raises(ValueError):
        _SliceTest()[1:]
    assert ae(_SliceTest()[:5], [0, 1, 2, 3, 4])
    assert ae(_SliceTest()[1:5], [1, 2, 3, 4])

    with raises(ValueError):
        _SliceTest()[::2]
    with raises(ValueError):
        _SliceTest()[1::2]
    assert ae(_SliceTest()[1:5:2], [1, 3])

    with raises(ValueError):
        _SliceTest(start=0)[:]
    with raises(ValueError):
        _SliceTest(start=1)[:]
    assert ae(_SliceTest(stop=5)[:], [0, 1, 2, 3, 4])


    # _range_from_slice(myslice, start=None, stop=None, length=None)
