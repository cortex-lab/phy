# -*- coding: utf-8 -*-

"""Test transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..transform import Translate, Scale, Range, Clip


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def _check(transform, array, expected):
    transformed = transform.apply(array)
    if array is None or not len(array):
        assert transformed == array
        return
    array = np.atleast_2d(array)
    if isinstance(array, np.ndarray):
        assert transformed.shape == array.shape
        assert transformed.dtype == np.float32
    assert np.all(transformed == expected)


#------------------------------------------------------------------------------
# Test transform
#------------------------------------------------------------------------------

def test_types():
    t = Translate([1, 2])
    _check(t, [], [])

    for ab in [[3, 4], [3., 4.]]:
        for arr in [ab, [ab], np.array(ab), np.array([ab]),
                    np.array([ab, ab, ab])]:
            _check(t, arr, [[4, 6]])


def test_translate():
    t = Translate([1, 2])
    _check(t, [3, 4], [[4, 6]])


def test_scale():
    t = Scale([-1, 2])
    _check(t, [3, 4], [[-3, 8]])


def test_range():
    t = Range([0, 1], [2, 3])

    # One element => move to the center of the window.
    _check(t, [-1, -1], [[1, 2]])
    _check(t, [3, 4], [[1, 2]])
    _check(t, [0, 1], [[1, 2]])

    # Extend the range symmetrically.
    _check(t, [[-1, 0], [3, 4]], [[0, 1], [2, 3]])


def test_clip():
    t = Clip([0, 1], [2, 3])

    _check(t, [-1, -1], [[0, 1]])
    _check(t, [3, 4], [[2, 3]])

    _check(t, [[-1, 0], [3, 4]], [[0, 1], [2, 3]])
