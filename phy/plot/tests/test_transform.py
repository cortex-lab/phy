# -*- coding: utf-8 -*-

"""Test transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent

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


def test_translate_numpy():
    t = Translate([1, 2])
    _check(t, [3, 4], [[4, 6]])


def test_scale_numpy():
    t = Scale([-1, 2])
    _check(t, [3, 4], [[-3, 8]])


def test_range_numpy():
    t = Range([0, 0, 1, 1], [-1, -1, 1, 1])

    _check(t, [-1, -1], [[-3, -3]])
    _check(t, [0, 0], [[-1, -1]])
    _check(t, [0.5, 0.5], [[0, 0]])
    _check(t, [1, 1], [[1, 1]])

    _check(t, [[0, .5], [1.5, -.5]], [[-1, 0], [2, -2]])


def test_clip_numpy():
    t = Clip([0, 1, 2, 3])

    _check(t, [-1, -1], [[0, 1]])
    _check(t, [3, 4], [[2, 3]])

    _check(t, [[-1, 0], [3, 4]], [[0, 1], [2, 3]])


#------------------------------------------------------------------------------
# Test GLSL transforms
#------------------------------------------------------------------------------

def test_translate_glsl():
    t = Translate('u_translate')
    assert t.glsl('x') == 'x + u_translate'


def test_scale_glsl():
    t = Scale('u_scale')
    assert t.glsl('x') == 'x * u_scale'


def test_range_glsl():
    t = Range(['u_from.xy', 'u_from.zw'], ['u_to.xy', 'u_to.zw'])
    assert t.glsl('x') == ('u_to.xy + (u_to.zw - u_to.xy) * (x - u_from.xy) / '
                           '(u_from.zw - u_from.xy)')


def test_clip_glsl():
    t = Clip(['xymin', 'xymax'])
    assert t.glsl('x') == dedent("""
        if ((x.x < xymin.x) |
            (x.y < xymin.y) |
            (x.x > xymax.x) |
            (x.y > xymax.y)) {
            discard;
        }
        """).strip()
