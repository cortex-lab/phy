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

def _wrap_apply(f, **kwargs_init):
    def wrapped(arr, **kwargs):
        if arr is None or not len(arr):
            return arr
        # Method kwargs first, then we update with the constructor kwargs.
        kwargs.update(kwargs_init)
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


def _wrap_glsl(f, **kwargs_init):
    def wrapped(var, **kwargs):
        # Method kwargs first, then we update with the constructor kwargs.
        kwargs.update(kwargs_init)
        out = f(var, **kwargs)
        out = dedent(out).strip()
        return out
    return wrapped


def _wrap(f, **kwargs_init):
    """Pass extra keyword arguments to a function.

    Used to pass constructor arguments to class methods in transforms.

    """
    def wrapped(*args, **kwargs):
        # Method kwargs first, then we update with the constructor kwargs.
        kwargs.update(kwargs_init)
        return f(*args, **kwargs)
    return wrapped


def _glslify(r):
    """Transform a string or a n-tuple to a valid GLSL expression."""
    if isinstance(r, string_types):
        return r
    else:
        assert 2 <= len(r) <= 4
        return 'vec{}({})'.format(len(r), ', '.join(map(str, r)))


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
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # Pass the constructor kwargs to the methods.
        self.apply = _wrap_apply(self.apply, **kwargs)
        self.glsl = _wrap_glsl(self.glsl, **kwargs)

    def apply(self, arr):
        raise NotImplementedError()

    def glsl(self, var):
        raise NotImplementedError()


class Translate(BaseTransform):
    def apply(self, arr, translate=None):
        assert isinstance(arr, np.ndarray)
        return arr + np.asarray(translate)

    def glsl(self, var, translate=None):
        assert var
        return """{var} = {var} + {translate};""".format(var=var,
                                                         translate=translate)


class Scale(BaseTransform):
    def apply(self, arr, scale=None):
        return arr * np.asarray(scale)

    def glsl(self, var, scale=None):
        assert var
        return """{var} = {var} * {scale};""".format(var=var, scale=scale)


class Range(BaseTransform):
    def apply(self, arr, from_bounds=None, to_bounds=NDC):
        f0 = np.asarray(from_bounds[:2])
        f1 = np.asarray(from_bounds[2:])
        t0 = np.asarray(to_bounds[:2])
        t1 = np.asarray(to_bounds[2:])

        return t0 + (t1 - t0) * (arr - f0) / (f1 - f0)

    def glsl(self, var, from_bounds=None, to_bounds=NDC):
        assert var

        from_bounds = _glslify(from_bounds)
        to_bounds = _glslify(to_bounds)

        return ("{var} = {t}.xy + ({t}.zw - {t}.xy) * "
                "({var} - {f}.xy) / ({f}.zw - {f}.xy);"
                "").format(var=var, f=from_bounds, t=to_bounds)


class Clip(BaseTransform):
    def apply(self, arr, bounds=NDC):
        index = ((arr[:, 0] >= bounds[0]) &
                 (arr[:, 1] >= bounds[1]) &
                 (arr[:, 0] <= bounds[2]) &
                 (arr[:, 1] <= bounds[3]))
        return arr[index, ...]

    def glsl(self, var, bounds=NDC):
        assert var

        bounds = _glslify(bounds)

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

    def __init__(self, **kwargs):
        super(Subplot, self).__init__(**kwargs)
        self.get_bounds = _wrap(self.get_bounds)

    def get_bounds(self, shape=None, index=None):
        return subplot_bounds(shape=shape, index=index)

    def apply(self, arr, shape=None, index=None):
        from_bounds = NDC
        to_bounds = self.get_bounds(shape=shape, index=index)
        return super(Subplot, self).apply(arr,
                                          from_bounds=from_bounds,
                                          to_bounds=to_bounds)

    def glsl(self, var, shape=None, index=None):
        assert var

        index = _glslify(index)
        shape = _glslify(shape)

        snippet = """
        float subplot_width = 2. / {shape}.y;
        float subplot_height = 2. / {shape}.x;

        float subplot_x = -1.0 + {index}.y * subplot_width;
        float subplot_y = +1.0 - ({index}.x + 1) * subplot_height;

        {var} = vec2(subplot_x + subplot_width * ({var}.x + 1) * .5,
                     subplot_y + subplot_height * ({var}.y + 1) * .5);
        """.format(index=index, shape=shape, var=var)

        snippet = snippet.format(index=index, shape=shape, var=var)
        return snippet


#------------------------------------------------------------------------------
# Transform chains
#------------------------------------------------------------------------------

class TransformChain(object):
    """A linear sequence of transforms that happen on the CPU and GPU."""
    def __init__(self, cpu_transforms=None, gpu_transforms=None):
        self.transformed_var_name = None
        self.cpu_transforms = []
        self.gpu_transforms = []
        self.add_cpu_transforms(cpu_transforms or [])
        self.add_gpu_transforms(gpu_transforms or [])

    def add_cpu_transforms(self, transforms):
        """Add some transforms."""
        self.cpu_transforms.extend(transforms or [])

    def add_gpu_transforms(self, transforms):
        """Add some transforms."""
        self.gpu_transforms.extend(transforms or [])

    def get(self, class_name):
        """Get a transform in the chain from its name."""
        for transform in self.cpu_transforms + self.gpu_transforms:
            if transform.__class__.__name__ == class_name:
                return transform

    def apply(self, arr):
        """Apply all CPU transforms on an array."""
        for t in self.cpu_transforms:
            arr = t.apply(arr)
        return arr
