# -*- coding: utf-8 -*-

"""Test plotting utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
from pytest import raises

from ..utils import (
    _load_shader, _tesselate_histogram, BatchAccumulator
)


#------------------------------------------------------------------------------
# Test utilities
#------------------------------------------------------------------------------

def test_load_shader():
    assert 'main()' in _load_shader('simple.vert')


def test_tesselate_histogram():
    n = 7
    hist = np.arange(n)
    thist = _tesselate_histogram(hist)
    assert thist.shape == (6 * n, 2)
    ac(thist[0], [0, 0])
    ac(thist[-3], [n, n - 1])
    ac(thist[-1], [n, 0])


def test_accumulator():
    b = BatchAccumulator()
    with raises(AttributeError):
        b.doesnotexist
    b.add({'x': np.ones(4), 'y': 2}, n_items=4)
    b.add({'x': np.zeros(2), 'y': 1}, n_items=2)
    x = np.array([[1, 1, 1, 1, 0, 0]]).T
    y = np.array([[2, 2, 2, 2, 1, 1]]).T
    ae(b.x, x)
    ae(b.y, y)
    assert tuple(b.data.keys()) == ('x', 'y')
    ae(b.data.x, x)
    ae(b.data.y, y)
