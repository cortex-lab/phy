# -*- coding: utf-8 -*-

"""Test transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent

import numpy as np
from numpy.testing import assert_equal as ae
from pytest import yield_fixture

from ..transform import (_glslify_range,
                         Translate, Scale, Range, Clip, Subplot, GPU,
                         TransformChain,
                         )


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
# Test utils
#------------------------------------------------------------------------------

def test_glslify_range():
    assert _glslify_range(('a', 'b')) == ('a', 'b')
    assert _glslify_range((1, 2, 3, 4)) == ('vec2(1.000, 2.000)',
                                            'vec2(3.000, 4.000)')


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


def test_translate_cpu():
    _check(Translate(translate=[1, 2]), [3, 4], [[4, 6]])


def test_scale_cpu():
    _check(Scale(), [3, 4], [[-3, 8]], scale=[-1, 2])


def test_range_cpu():
    kwargs = dict(from_range=[0, 0, 1, 1], to_range=[-1, -1, 1, 1])

    _check(Range(), [-1, -1], [[-3, -3]], **kwargs)
    _check(Range(), [0, 0], [[-1, -1]], **kwargs)
    _check(Range(), [0.5, 0.5], [[0, 0]], **kwargs)
    _check(Range(), [1, 1], [[1, 1]], **kwargs)

    _check(Range(), [[0, .5], [1.5, -.5]], [[-1, 0], [2, -2]], **kwargs)


def test_clip_cpu():
    kwargs = dict(bounds=[0, 1, 2, 3])

    _check(Clip(), [0, 0], [0, 0])  # Default bounds.

    _check(Clip(), [0, 1], [0, 1], **kwargs)
    _check(Clip(), [1, 2], [1, 2], **kwargs)
    _check(Clip(), [2, 3], [2, 3], **kwargs)

    _check(Clip(), [-1, -1], [], **kwargs)
    _check(Clip(), [3, 4], [], **kwargs)
    _check(Clip(), [[-1, 0], [3, 4]], [], **kwargs)


def test_subplot_cpu():
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

    assert Range(from_range=[-1, -1, 1, 1]).glsl('x')

    expected = ('u_to.xy + (u_to.zw - u_to.xy) * (x - u_from.xy) / '
                '(u_from.zw - u_from.xy)')
    r = Range(to_range=['u_to.xy', 'u_to.zw'])
    assert expected in r.glsl('x', from_range=['u_from.xy', 'u_from.zw'])


def test_clip_glsl():
    expected = dedent("""
        if ((x.x < xymin.x) ||
            (x.y < xymin.y) ||
            (x.x > xymax.x) ||
            (x.y > xymax.y)) {
            discard;
        }
        """).strip()
    assert expected in Clip().glsl('x', bounds=['xymin', 'xymax'])


def test_subplot_glsl():
    glsl = Subplot().glsl('x', shape='u_shape', index='a_index')
    assert 'x = ' in glsl


#------------------------------------------------------------------------------
# Test transform chain
#------------------------------------------------------------------------------

@yield_fixture
def array():
    yield np.array([[-1, 0], [1, 2]])


def test_transform_chain_empty(array):
    t = TransformChain([])

    assert t.cpu_transforms == []
    assert t.gpu_transforms == []
    assert t.get('GPU') is None

    ae(t.apply(array), array)


def test_transform_chain_one(array):
    translate = Translate(translate=[1, 2])
    t = TransformChain([translate])

    assert t.cpu_transforms == [translate]
    assert t.gpu_transforms == []

    ae(t.apply(array), [[0, 2], [2, 4]])


def test_transform_chain_two(array):
    translate = Translate(translate=[1, 2])
    scale = Scale(scale=[.5, .5])
    t = TransformChain([translate, scale])

    assert t.cpu_transforms == [translate, scale]
    assert t.gpu_transforms == []

    assert isinstance(t.get('Translate'), Translate)
    assert t.get('Unknown') is None

    ae(t.apply(array), [[0, 1], [1, 2]])


def test_transform_chain_complete(array):
    t = TransformChain([Scale(scale=.5),
                        Scale(scale=2.)])
    t.add([Range(from_range=[-3, -3, 1, 1]),
           GPU(),
           Clip(),
           Subplot(shape='u_shape', index='a_box'),
           ])

    assert len(t.cpu_transforms) == 3
    assert len(t.gpu_transforms) == 2

    ae(t.apply(array), [[0, .5], [1, 1.5]])

    vs = dedent("""
    attribute vec2 a_position;
    void main() {
        gl_Position = transform(a_position);
    }
    """).strip()

    fs = dedent("""
    void main() {
        gl_FragColor = vec4(1., 1., 1., 1.);
    }
    """).strip()
    vs, fs = t.insert_glsl(vs, fs)
    assert 'a_box' in vs
    assert 'v_' in vs
    assert 'v_' in fs
    assert 'discard' in fs

    # Increase coverage.
    t.insert_glsl(vs.replace('transform', ''), fs)
