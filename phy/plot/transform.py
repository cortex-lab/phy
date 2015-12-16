# -*- coding: utf-8 -*-

"""Transforms."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent

import numpy as np
from six import string_types

import logging

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _wrap_apply(f):
    def wrapped(arr, **kwargs):
        if arr is None or not len(arr):
            return arr
        arr = np.atleast_2d(arr)
        arr = arr.astype(np.float32)
        assert arr.ndim == 2
        out = f(arr, **kwargs)
        out = out.astype(np.float32)
        out = np.atleast_2d(out)
        assert out.ndim == 2
        assert out.shape[1] == arr.shape[1]
        return out
    return wrapped


def _wrap_glsl(f):
    def wrapped(var, **kwargs):
        out = f(var, **kwargs)
        out = dedent(out).strip()
        return out
    return wrapped


def _glslify(r):
    """Transform a string or a n-tuple to a valid GLSL expression."""
    if isinstance(r, string_types):
        return r
    else:
        assert 2 <= len(r) <= 4
        return 'vec{}({})'.format(len(r), ', '.join(map(str, r)))


def _minus(value):
    if isinstance(value, np.ndarray):
        return -value
    else:
        assert len(value) == 2
        return -value[0], -value[1]


def _inverse(value):
    if isinstance(value, np.ndarray):
        return 1. / value
    else:
        assert len(value) == 2
        return 1. / value[0], 1. / value[1]


def subplot_bounds(shape=None, index=None):
    i, j = index
    n_rows, n_cols = shape

    assert 0 <= i <= n_rows - 1
    assert 0 <= j <= n_cols - 1

    width = 2.0 / n_cols
    height = 2.0 / n_rows

    x = -1.0 + j * width
    y = +1.0 - (i + 1) * height

    return [x, y, x + width, y + height]


def subplot_bounds_glsl(shape=None, index=None):
    x0 = '-1.0 + 2.0 * {i}.y / {s}.y'.format(s=shape, i=index)
    y0 = '+1.0 - 2.0 * ({i}.x + 1) / {s}.x'.format(s=shape, i=index)
    x1 = '-1.0 + 2.0 * ({i}.y + 1) / {s}.y'.format(s=shape, i=index)
    y1 = '+1.0 - 2.0 * ({i}.x) / {s}.x'.format(s=shape, i=index)
    return 'vec4({x0}, {y0}, {x1}, {y1})'.format(x0=x0, y0=y0, x1=x1, y1=y1)


def pixels_to_ndc(pos, size=None):
    """Convert from pixels to normalized device coordinates (in [-1, 1])."""
    pos = np.asarray(pos, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    pos = pos / (size / 2.) - 1
    # Flip y, because the origin in pixels is at the top left corner of the
    # window.
    pos[1] = -pos[1]
    return pos


"""Bounds in Normalized Device Coordinates (NDC)."""
NDC = (-1.0, -1.0, +1.0, +1.0)


#------------------------------------------------------------------------------
# Transforms
#------------------------------------------------------------------------------

class BaseTransform(object):
    def __init__(self, value=None):
        self.value = value
        self.apply = _wrap_apply(self.apply)
        self.glsl = _wrap_glsl(self.glsl)

    def apply(self, arr):
        raise NotImplementedError()

    def glsl(self, var):
        raise NotImplementedError()

    def inverse(self):
        raise NotImplementedError()


class Translate(BaseTransform):
    def apply(self, arr):
        assert isinstance(arr, np.ndarray)
        return arr + np.asarray(self.value)

    def glsl(self, var):
        assert var
        return """{var} = {var} + {translate};""".format(var=var,
                                                         translate=self.value)

    def inverse(self):
        if isinstance(self.value, string_types):
            return Translate('-' + self.value)
        else:
            return Translate(_minus(self.value))


class Scale(BaseTransform):
    def apply(self, arr):
        return arr * np.asarray(self.value)

    def glsl(self, var):
        assert var
        return """{var} = {var} * {scale};""".format(var=var, scale=self.value)

    def inverse(self):
        if isinstance(self.value, string_types):
            return Scale('1.0 / ' + self.value)
        else:
            return Scale(_inverse(self.value))


class Range(BaseTransform):
    def __init__(self, from_bounds=None, to_bounds=None):
        super(Range, self).__init__()
        self.from_bounds = from_bounds if from_bounds is not None else NDC
        self.to_bounds = to_bounds if to_bounds is not None else NDC

    def apply(self, arr):
        self.from_bounds = np.asarray(self.from_bounds)
        self.to_bounds = np.asarray(self.to_bounds)

        f0 = np.asarray(self.from_bounds[..., :2])
        f1 = np.asarray(self.from_bounds[..., 2:])
        t0 = np.asarray(self.to_bounds[..., :2])
        t1 = np.asarray(self.to_bounds[..., 2:])

        return t0 + (t1 - t0) * (arr - f0) / (f1 - f0)

    def glsl(self, var):
        assert var

        from_bounds = _glslify(self.from_bounds)
        to_bounds = _glslify(self.to_bounds)

        return ("{var} = {t}.xy + ({t}.zw - {t}.xy) * "
                "({var} - {f}.xy) / ({f}.zw - {f}.xy);"
                "").format(var=var, f=from_bounds, t=to_bounds)

    def inverse(self):
        return Range(from_bounds=self.to_bounds,
                     to_bounds=self.from_bounds)


class Clip(BaseTransform):
    def __init__(self, bounds=None):
        super(Clip, self).__init__()
        self.bounds = bounds or NDC

    def apply(self, arr):
        index = ((arr[:, 0] >= self.bounds[0]) &
                 (arr[:, 1] >= self.bounds[1]) &
                 (arr[:, 0] <= self.bounds[2]) &
                 (arr[:, 1] <= self.bounds[3]))
        return arr[index, ...]

    def glsl(self, var):
        assert var
        bounds = _glslify(self.bounds)

        return """
            if (({var}.x < {bounds}.x) ||
                ({var}.y < {bounds}.y) ||
                ({var}.x > {bounds}.z) ||
                ({var}.y > {bounds}.w)) {{
                discard;
            }}
        """.format(bounds=bounds, var=var)


class Subplot(Range):
    """Assume that the from_bounds is [-1, -1, 1, 1]."""

    def __init__(self, shape, index=None):
        super(Subplot, self).__init__()
        self.shape = shape
        self.index = index
        self.from_bounds = NDC
        if isinstance(self.shape, tuple) and isinstance(self.index, tuple):
            self.to_bounds = subplot_bounds(shape=self.shape, index=self.index)
        elif (isinstance(self.shape, string_types) and
                isinstance(self.index, string_types)):
            self.to_bounds = subplot_bounds_glsl(shape=self.shape,
                                                 index=self.index)


#------------------------------------------------------------------------------
# Transform chains
#------------------------------------------------------------------------------

class TransformChain(object):
    """A linear sequence of transforms that happen on the CPU and GPU."""
    def __init__(self):
        self.transformed_var_name = None
        self.cpu_transforms = []
        self.gpu_transforms = []

    def add_on_cpu(self, transforms):
        """Add some transforms."""
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.cpu_transforms.extend(transforms or [])
        return self

    def add_on_gpu(self, transforms):
        """Add some transforms."""
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.gpu_transforms.extend(transforms or [])
        return self

    def get(self, class_name):
        """Get a transform in the chain from its name."""
        for transform in self.cpu_transforms + self.gpu_transforms:
            if transform.__class__.__name__ == class_name:
                return transform

    def _remove_transform(self, transforms, name):
        return [t for t in transforms if t.__class__.__name__ != name]

    def remove(self, name):
        """Remove a transform in the chain."""
        cpu_transforms = self._remove_transform(self.cpu_transforms, name)
        gpu_transforms = self._remove_transform(self.gpu_transforms, name)
        return (TransformChain().add_on_cpu(cpu_transforms).
                add_on_gpu(gpu_transforms))

    def apply(self, arr):
        """Apply all CPU transforms on an array."""
        for t in self.cpu_transforms:
            arr = t.apply(arr)
        return arr

    def __add__(self, tc):
        assert isinstance(tc, TransformChain)
        assert tc.transformed_var_name == self.transformed_var_name
        self.cpu_transforms.extend(tc.cpu_transforms)
        self.gpu_transforms.extend(tc.gpu_transforms)
        return self
