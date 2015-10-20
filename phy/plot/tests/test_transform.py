# -*- coding: utf-8 -*-

"""Test transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent

import numpy as np

from ..transform import Translate, Scale, Range, Clip, Subplot


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def _check(transform, array, expected, **kwargs):
    transformed = transform.apply(array, **kwargs)
    if array is None or not len(array):
        assert transformed == array
        return
    array = np.atleast_2d(array)
    if isinstance(array, np.ndarray):
        assert transformed.shape[1] == array.shape[1]
        assert transformed.dtype == np.float32
    if not len(transformed):
        assert not len(expected)
    else:
        assert np.allclose(transformed, expected)


#------------------------------------------------------------------------------
# Test transform
#------------------------------------------------------------------------------

def test_types():
    t = Translate()
    _check(t, [], [], translate=[1, 2])

    for ab in [[3, 4], [3., 4.]]:
        for arr in [ab, [ab], np.array(ab), np.array([ab]),
                    np.array([ab, ab, ab])]:
            _check(t, arr, [[4, 6]], translate=[1, 2])


def test_translate_numpy():
    _check(Translate(translate=[1, 2]), [3, 4], [[4, 6]])


def test_scale_numpy():
    _check(Scale(), [3, 4], [[-3, 8]], scale=[-1, 2])


def test_range_numpy():
    kwargs = dict(from_range=[0, 0, 1, 1], to_range=[-1, -1, 1, 1])

    _check(Range(), [-1, -1], [[-3, -3]], **kwargs)
    _check(Range(), [0, 0], [[-1, -1]], **kwargs)
    _check(Range(), [0.5, 0.5], [[0, 0]], **kwargs)
    _check(Range(), [1, 1], [[1, 1]], **kwargs)

    _check(Range(), [[0, .5], [1.5, -.5]], [[-1, 0], [2, -2]], **kwargs)


def test_clip_numpy():
    kwargs = dict(bounds=[0, 1, 2, 3])

    _check(Clip(), [0, 1], [0, 1], **kwargs)
    _check(Clip(), [1, 2], [1, 2], **kwargs)
    _check(Clip(), [2, 3], [2, 3], **kwargs)

    _check(Clip(), [-1, -1], [], **kwargs)
    _check(Clip(), [3, 4], [], **kwargs)
    _check(Clip(), [[-1, 0], [3, 4]], [], **kwargs)


def test_subplot_numpy():
    shape = (2, 3)

    _check(Subplot(), [-1, -1], [-1, +0], index=(0, 0), shape=shape)
    _check(Subplot(), [+0, +0], [-2. / 3., .5], index=(0, 0), shape=shape)

    _check(Subplot(), [-1, -1], [-1, -1], index=(1, 0), shape=shape)
    _check(Subplot(), [+1, +1], [-1. / 3, 0], index=(1, 0), shape=shape)

    _check(Subplot(), [0, 1], [0, 0], index=(1, 1), shape=shape)


#------------------------------------------------------------------------------
# Test GLSL transforms
#------------------------------------------------------------------------------

def test_translate_glsl():
    t = Translate(translate='u_translate').glsl('x')
    assert 'x = x + u_translate' in t


def test_scale_glsl():
    assert 'x = x * u_scale' in Scale().glsl('x', scale='u_scale')


def test_range_glsl():
    expected = ('u_to.xy + (u_to.zw - u_to.xy) * (x - u_from.xy) / '
                '(u_from.zw - u_from.xy)')
    assert expected in Range().glsl('x',
                                    from_range=['u_from.xy', 'u_from.zw'],
                                    to_range=['u_to.xy', 'u_to.zw'])


def test_clip_glsl():
    expected = dedent("""
        if ((x.x < xymin.x) |
            (x.y < xymin.y) |
            (x.x > xymax.x) |
            (x.y > xymax.y)) {
            discard;
        }
        """).strip()
    assert expected in Clip().glsl('x', bounds=['xymin', 'xymax'])


def test_subplot_glsl():
    glsl = Subplot().glsl('x', shape='u_shape', index='a_index')
    assert 'x = ' in glsl
