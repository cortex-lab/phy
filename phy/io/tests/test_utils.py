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
    assert ae(_SliceTest()[1:5], [1, 2, 3, 4])
    # _range_from_slice(myslice, start=None, stop=None, length=None)
