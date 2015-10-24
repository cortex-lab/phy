# -*- coding: utf-8 -*-

"""Transforms."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent
import re

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
    def wrapped(*args, **kwargs):
        # Method kwargs first, then we update with the constructor kwargs.
        kwargs.update(kwargs_init)
        return f(*args, **kwargs)
    return wrapped


def indent(text):
    return '\n'.join('    ' + l.strip() for l in text.splitlines())


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
        self.pre_transforms = _wrap(self.pre_transforms, **kwargs)
        self.post_transforms = _wrap(self.post_transforms, **kwargs)

    def pre_transforms(self, **kwargs):
        return []

    def post_transforms(self, **kwargs):
        return []

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

class GPU(object):
    """Used to specify that the next transforms in the chain happen on
    the GPU."""
    pass


class TransformChain(object):
    """A linear sequence of transforms that happen on the CPU and GPU."""
    def __init__(self, transforms=None):
        self.transformed_var_name = None
        self.transforms = []
        self.add(transforms)

    def _index_of_gpu(self):
        classes = [t.__class__.__name__ for t in self.transforms]
        return classes.index('GPU') if 'GPU' in classes else None

    @property
    def cpu_transforms(self):
        """All transforms until `GPU()`."""
        i = self._index_of_gpu()
        return self.transforms[:i] if i is not None else self.transforms

    @property
    def gpu_transforms(self):
        """All transforms after `GPU()`."""
        i = self._index_of_gpu()
        return self.transforms[i + 1:] if i is not None else []

    def add(self, transforms):
        """Add some transforms."""
        for t in transforms:
            if hasattr(t, 'pre_transforms'):
                for p in t.pre_transforms():
                    self.transforms.append(p)
            self.transforms.append(t)
            if hasattr(t, 'post_transforms'):
                for p in t.post_transforms():
                    self.transforms.append(p)

    def get(self, class_name):
        """Get a transform in the chain from its name."""
        for transform in self.transforms:
            if transform.__class__.__name__ == class_name:
                return transform

    def apply(self, arr):
        """Apply all CPU transforms on an array."""
        for t in self.cpu_transforms:
            arr = t.apply(arr)
        return arr

    def insert_glsl(self, vertex, fragment):
        """Generate the GLSL code of the transform chain."""

        # Find the place where to insert the GLSL snippet.
        # This is "gl_Position = transform(data_var_name);" where
        # data_var_name is typically an attribute.
        vs_regex = re.compile(r'gl_Position = transform\(([\S]+)\);')
        r = vs_regex.search(vertex)
        if not r:
            logger.debug("The vertex shader doesn't contain the transform "
                         "placeholder: skipping the transform chain "
                         "GLSL insertion.")
            return vertex, fragment
        assert r
        logger.log(5, "Found transform placeholder in vertex code: `%s`",
                   r.group(0))

        # Find the GLSL variable with the data (should be a `vec2`).
        var = r.group(1)
        self.transformed_var_name = var
        assert var and var in vertex

        # Generate the snippet to insert in the shaders.
        temp_var = 'temp_pos_tr'
        # Name for the (eventual) varying.
        fvar = 'v_{}'.format(temp_var)
        vs_insert = "vec2 {} = {};\n".format(temp_var, var)
        for t in self.gpu_transforms:
            if isinstance(t, Clip):
                # Set the varying value in the vertex shader.
                vs_insert += '{} = {};\n'.format(fvar, temp_var)
                continue
            vs_insert += t.glsl(temp_var) + '\n'
        vs_insert += 'gl_Position = vec4({}, 0., 1.);\n'.format(temp_var)

        # Clipping.
        clip = self.get('Clip')
        if clip:
            # Varying name.
            glsl_clip = clip.glsl(fvar)

            # Prepare the fragment regex.
            fs_regex = re.compile(r'(void main\(\)\s*\{)')
            fs_insert = '\\1\n{}'.format(glsl_clip)

            # Add the varying declaration for clipping.
            varying_decl = 'varying vec2 {};\n'.format(fvar)
            vertex = varying_decl + vertex
            fragment = varying_decl + fragment

            # Make the replacement in the fragment shader for clipping.
            fragment = fs_regex.sub(indent(fs_insert), fragment)

        # Insert the GLSL snippet of the transform chain in the vertex shader.
        vertex = vs_regex.sub(indent(vs_insert), vertex)

        return vertex, fragment
