# -*- coding: utf-8 -*-

"""Test transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent

import numpy as np
from numpy.testing import assert_equal as ae
from numpy.testing import assert_allclose as ac
from pytest import fixture

from ..transform import (
    _glslify, pixels_to_ndc, _normalize, extend_bounds,
    Translate, Scale, Rotate, Range, Clip, Subplot, TransformChain)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def _check_forward(transform, array, expected):
    transformed = transform.apply(array)
    if array is None or not len(array):
        assert transformed is None or not len(transformed)
        return
    array = np.atleast_2d(array)
    if isinstance(array, np.ndarray):
        assert transformed.shape[1] == array.shape[1]
    if not len(transformed):
        assert not len(expected)
    else:
        assert np.allclose(transformed, expected, atol=1e-7)


def _check(transform, array, expected):
    array = np.array(array, dtype=np.float64)
    expected = np.array(expected, dtype=np.float64)
    _check_forward(transform, array, expected)
    # Test the inverse transform if it is implemented.
    inv = transform.inverse()
    _check_forward(inv, expected, array)


#------------------------------------------------------------------------------
# Test utils
#------------------------------------------------------------------------------

def test_glslify():
    assert _glslify('a') == 'a', 'b'
    assert _glslify((1, 2, 3, 4)) == 'vec4(1, 2, 3, 4)'
    assert _glslify((1., 2.)) == 'vec2(1.0, 2.0)'


def test_pixels_to_ndc():
    assert list(pixels_to_ndc((0, 0), size=(10, 10))) == [-1, 1]


def test_normalize():
    m, M = 0., 10.
    arr = np.linspace(0., 10., 10)
    ac(_normalize(arr, m, M), np.linspace(-1., 1., 10))
    ac(_normalize(arr, m, m), arr)


def test_extend_bounds():
    assert extend_bounds([(0, 0, 1, 1), (-1, -2, 3, 4)]) == (-1, -2, 3, 4)
    assert extend_bounds([(0, 0, 0, 0)]) == (-1, -1, 1, 1)


#------------------------------------------------------------------------------
# Test transform
#------------------------------------------------------------------------------

def test_types():
    _check(Translate([1, 2]), [], [])

    for ab in [[3, 4], [3., 4.]]:
        for arr in [ab, [ab], np.array(ab), np.array([ab]),
                    np.array([ab, ab, ab])]:
            _check(Translate([1, 2]), arr, [[4, 6]])


def test_translate_cpu():
    _check(Translate([1, 2]), [3, 4], [[4, 6]])
    _check(Translate([1, 2]).inverse(), [4, 6], [[3, 4]])
    _check(Translate(np.array([[1, 2]])).inverse(), [4, 6], [[3, 4]])

    # Callable.
    _check(Translate(lambda: [1, 2]), [3, 4], [[4, 6]])


def test_scale_cpu():
    _check(Scale([-1, 2]), [3, 4], [[-3, 8]])
    _check(Scale([-1, 2]).inverse(), [-3, 8], [[3, 4]])
    _check(Scale(np.array([[-1, 2]])).inverse(), [-3, 8], [[3, 4]])


def test_rotate_cpu():
    _check(Rotate(), [1, 0], [[0, -1]])
    _check(Rotate('ccw'), [0, 1], [[-1, 0]])
    _check(Rotate().inverse(), [1, 1], [[-1, 1]])


def test_range_cpu():
    _check(Range([0, 0, 1, 1], [-1, -1, 1, 1]), [-1, -1], [[-3, -3]])
    _check(Range([0, 0, 1, 1], [-1, -1, 1, 1]), [0, 0], [[-1, -1]])
    _check(Range([0, 0, 1, 1], [-1, -1, 1, 1]), [0.5, 0.5], [[0, 0]])
    _check(Range([0, 0, 1, 1], [-1, -1, 1, 1]), [1, 1], [[1, 1]])

    _check(Range([0, 0, 1, 1], [-1, -1, 1, 1]),
           [[0, .5], [1.5, -.5]], [[-1, 0], [2, -2]])


def test_range_cpu_vectorized():
    arr = np.arange(6).reshape((3, 2)) * 1.
    arr_tr = arr / 5.
    arr_tr[2, :] /= 10

    f = np.tile([0, 0, 5, 5], (3, 1))
    f[2, :] *= 10

    t = np.tile([0, 0, 1, 1], (3, 1))

    _check(Range(f, t), arr, arr_tr)


def test_clip_cpu():
    _check(Clip(), [0, 0], [0, 0])  # Default bounds.

    _check(Clip([0, 1, 2, 3]), [0, 1], [0, 1])
    _check(Clip([0, 1, 2, 3]), [1, 2], [1, 2])
    _check(Clip([0, 1, 2, 3]), [2, 3], [2, 3])

    _check(Clip([0, 1, 2, 3]), [-1, -1], [])
    _check(Clip([0, 1, 2, 3]), [3, 4], [])
    _check(Clip([0, 1, 2, 3]), [[-1, 0], [3, 4]], [])


def test_subplot_cpu():
    shape = (2, 3)

    _check(Subplot(shape, (0, 0)), [-1, -1], [-1, +0])
    _check(Subplot(shape, (0, 0)), [+0, +0], [-2. / 3., .5])

    _check(Subplot(shape, (1, 0)), [-1, -1], [-1, -1])
    _check(Subplot(shape, (1, 0)), [+1, +1], [-1. / 3, 0])

    _check(Subplot(shape, (1, 1)), [0, 1], [0, 0])


#------------------------------------------------------------------------------
# Test GLSL transforms
#------------------------------------------------------------------------------

def test_translate_glsl():
    assert 'x = x + u_translate' in Translate(gpu_var='u_translate').glsl('x')
    assert 'x + -u_translate' in Translate(gpu_var='u_translate').inverse().glsl('x')


def test_scale_glsl():
    assert 'x = x * u_scale' in Scale(gpu_var='u_scale').glsl('x')
    assert 'x = x * 1.0 / u_scale' in Scale(gpu_var='u_scale').inverse().glsl('x')


def test_rotate_glsl():
    assert 'pos = vec2(-pos.y, pos.x)' in Rotate('ccw').glsl('pos')
    assert 'pos = -vec2(-pos.y, pos.x)' in Rotate('cw').glsl('pos')


def test_range_glsl():

    assert Range([-1, -1, 1, 1]).glsl('x')
    r = Range('u_from', 'u_to')
    assert 'x = (x - ' in r.glsl('x')
    assert 'u_from' in r.glsl('x')


def test_clip_glsl():
    expected = dedent("""
        if ((x.x < b.x) ||
            (x.y < b.y) ||
            (x.x > b.z) ||
            (x.y > b.w)) {
            discard;
        }
        """).strip()
    assert expected in Clip('b').glsl('x')


def test_subplot_glsl():
    glsl = Subplot('u_shape', 'a_index').glsl('x')
    assert 'x = ' in glsl


#------------------------------------------------------------------------------
# Test transform chain
#------------------------------------------------------------------------------

@fixture
def array():
    return np.array([[-1., 0.], [1., 2.]])


def test_transform_chain_empty(array):
    t = TransformChain()

    assert t.transforms == []

    ae(t.apply(array), array)


def test_transform_chain_one(array):
    translate = Translate([1, 2])
    t = TransformChain()
    t.add([translate])

    assert t.transforms == [translate]

    ae(t.apply(array), [[0, 2], [2, 4]])


def test_transform_chain_two(array):
    translate = Translate([1, 2])
    scale = Scale([.5, .5])
    t = TransformChain([translate, scale])

    assert t.transforms == [translate, scale]
    assert t[0] == translate
    assert t[1] == scale

    assert isinstance(t.get('Translate'), Translate)
    assert t.get('Unknown') is None

    ae(t.apply(array), [[0, 1], [1, 2]])


def test_transform_chain_complete(array):
    t = Scale(.5) + Scale(2.) + Range([-3, -3, 1, 1]) + Subplot('u_shape', 'a_box_index')
    assert len(t.transforms) == 4
    ae(t.apply(array), [[0, .5], [1, 1.5]])


def test_transform_chain_add():
    tc = TransformChain()
    tc.add([Scale(.5)])

    tc_2 = TransformChain()
    tc_2.add([Scale(2.)])

    ae((tc + tc_2).apply([3.]), [[3.]])

    assert str(tc)


def test_transform_chain_inverse():
    tc = TransformChain()
    tc.add([Scale(.5), Translate((1, 0)), Scale(2)])
    tci = tc.inverse()
    ae(tc.apply([[1., 0.]]), [[3., 0.]])
    ae(tci.apply([[3., 0.]]), [[1., 0.]])
