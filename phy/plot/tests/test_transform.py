# -*- coding: utf-8 -*-

"""Test transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from itertools import product

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import yield_fixture

from ..transform import Translate, Scale, Range, Clip, GPU


#------------------------------------------------------------------------------
# Test transform
#------------------------------------------------------------------------------

@yield_fixture(params=product([0, 1, 2], ['i', 'f']))
def array(request):
    m, t = request.param
    if t == 'i':
        a, b = 3, 4
    elif t == 'f':
        a, b = 3., 4.
    arr = [a, b]
    if m == 1:
        arr = [arr]
    elif m == 2:
        arr = np.array(arr)
    elif m == 3:
        arr = np.array([arr])
    elif m == 4:
        arr = np.array([arr, arr, arr])
    yield arr


def _check(transform, array, expected):
    transformed = transform.apply(array)
    array = np.atleast_2d(array)
    if isinstance(array, np.ndarray):
        assert transformed.shape == array.shape
        assert transformed.dtype == array.dtype
    ae(transformed, expected)


def test_translate(array):
    t = Translate(1, 2)
    _check(t, array, [[4, 6]])
