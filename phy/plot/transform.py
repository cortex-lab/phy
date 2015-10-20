# -*- coding: utf-8 -*-

"""Transforms."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent
import re

import numpy as np

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


def indent(text):
    return '\n'.join('    ' + l.strip() for l in text.splitlines())


#------------------------------------------------------------------------------
# Transforms
#------------------------------------------------------------------------------

class BaseTransform(object):
    def __init__(self, **kwargs):
        # Pass the constructor kwargs to the methods.
        self.apply = _wrap_apply(self.apply, **kwargs)
        self.glsl = _wrap_glsl(self.glsl, **kwargs)

    def apply(self, arr):
        raise NotImplementedError()

    def glsl(self, var):
        raise NotImplementedError()


class Translate(BaseTransform):
    def apply(self, arr, translate=None):
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
    def apply(self, arr, from_range=None, to_range=None):
        if to_range is None:
            to_range = [-1, -1, 1, 1]

        f0 = np.asarray(from_range[:2])
        f1 = np.asarray(from_range[2:])
        t0 = np.asarray(to_range[:2])
        t1 = np.asarray(to_range[2:])

        return t0 + (t1 - t0) * (arr - f0) / (f1 - f0)

    def glsl(self, var, from_range=None, to_range=None):
        assert var
        if to_range is None:
            to_range = [-1, -1, 1, 1]

        return """
            {var} = {t0} + ({t1} - {t0}) * ({var} - {f0}) / ({f1} - {f0});
        """.format(var=var,
                   f0=from_range[0], f1=from_range[1],
                   t0=to_range[0], t1=to_range[1],
                   )


class Clip(BaseTransform):
    def apply(self, arr, bounds=None):
        if bounds is None:
            bounds = [-1, -1, 1, 1]

        xymin = np.asarray(bounds[:2])
        xymax = np.asarray(bounds[2:])
        index = ((arr[:, 0] >= xymin[0]) &
                 (arr[:, 1] >= xymin[1]) &
                 (arr[:, 0] <= xymax[0]) &
                 (arr[:, 1] <= xymax[1]))
        return arr[index, ...]

    def glsl(self, var, bounds=None):
        assert var
        if bounds is None:
            bounds = 'vec2(-1, -1)', 'vec2(1, 1)'

        return """
            if (({var}.x < {xymin}.x) |
                ({var}.y < {xymin}.y) |
                ({var}.x > {xymax}.x) |
                ({var}.y > {xymax}.y)) {{
                discard;
            }}
        """.format(xymin=bounds[0],
                   xymax=bounds[1],
                   var=var,
                   )


class Subplot(Range):
    """Assume that the from range is [-1, -1, 1, 1]."""
    def apply(self, arr, shape=None, index=None):
        i, j = index
        n_rows, n_cols = shape
        assert 0 <= i <= n_rows - 1
        assert 0 <= j <= n_cols - 1

        x = -1.0 + j * (2.0 / n_cols)
        y = +1.0 - i * (2.0 / n_rows)

        width = 2.0 / n_cols
        height = 2.0 / n_rows

        # The origin (x, y) corresponds to the lower-left corner of the
        # target box.
        y -= height

        from_range = [-1, -1, 1, 1]
        to_range = [x, y, x + width, y + height]

        return super(Subplot, self).apply(arr,
                                          from_range=from_range,
                                          to_range=to_range)

    def glsl(self, var, shape=None, index=None):
        assert var
        glsl = """
        float subplot_x = -1.0 + {index}.y * 2.0 / {shape}.y;
        float subplot_y = +1.0 - {index}.x * 2.0 / {shape}.x;

        float subplot_width = 2. / {shape}.y;
        float subplot_height = 2. / {shape}.x;

        {var} = vec2(subplot_x + subplot_width * {var}.x,
                     subplot_y + subplot_height * {var}.y);
        """
        return glsl.format(index=index, shape=shape, var=var)


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
        self.transforms = transforms or []

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
        self.transforms.extend(transforms)

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
        assert r, ("The vertex shader must contain the transform placeholder.")
        logger.debug("Found transform placeholder in vertex code: `%s`",
                     r.group(0))

        # Find the GLSL variable with the data (should be a `vec2`).
        var = r.group(1)
        assert var and var in vertex

        # Generate the snippet to insert in the shaders.
        vs_insert = ""
        for t in self.gpu_transforms:
            if isinstance(t, Clip):
                continue
            vs_insert += t.glsl(var) + '\n'
        vs_insert += 'gl_Position = {};\n'.format(var)

        # Clipping.
        clip = self.get('Clip')
        if clip:
            # Varying name.
            fvar = 'v_{}'.format(var)
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
            # Set the varying value in the vertex shader.
            vs_insert += '{} = {};\n'.format(fvar, var)

        # Insert the GLSL snippet of the transform chain in the vertex shader.
        vertex = vs_regex.sub(indent(vs_insert), vertex)

        return vertex, fragment
