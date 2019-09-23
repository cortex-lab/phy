# -*- coding: utf-8 -*-

"""Transforms."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from textwrap import dedent

import numpy as np

from phylib.utils.geometry import range_transform

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _wrap_apply(f):
    """Validate the input and output of transform apply functions."""
    def wrapped(arr, **kwargs):
        if arr is None or not len(arr):
            return arr
        arr = np.atleast_2d(arr)
        assert arr.ndim == 2
        assert arr.dtype in (np.float32, np.float64)
        out = f(arr, **kwargs)
        assert out.dtype == arr.dtype
        out = np.atleast_2d(out)
        assert out.ndim == 2
        assert out.shape[1] == arr.shape[1]
        return out
    return wrapped


def _wrap_glsl(f):
    """Validate the output of GLSL functions."""
    def wrapped(var, **kwargs):
        out = f(var, **kwargs)
        out = dedent(out).strip()
        return out
    return wrapped


def _glslify(r):
    """Transform a string or a n-tuple to a valid GLSL expression."""
    if isinstance(r, str):
        return r
    else:
        r = _call_if_callable(r)
        assert 2 <= len(r) <= 4
        return 'vec{}({})'.format(len(r), ', '.join(map(str, r)))


def _call_if_callable(s):
    """Call a variable if it's a callable, otherwise return it."""
    if hasattr(s, '__call__'):
        return s()
    return s


def _minus(value):
    if isinstance(value, np.ndarray):
        return -value
    else:
        assert len(value) == 2
        return -value[0], -value[1]


def _inverse(value):
    if isinstance(value, np.ndarray):
        return 1. / value
    elif hasattr(value, '__len__'):
        assert len(value) == 2
        return 1. / value[0], 1. / value[1]
    else:
        return 1. / value


def _normalize(arr, m, M):
    d = float(M - m)
    if abs(d) < 1e-9:
        return arr
    b = 2. / d
    a = -1 - 2. * m / d
    arr *= b
    arr += a
    return arr


def _fix_coordinate_in_visual(visual, coord):
    """Insert GLSL code to fix the position on the x or y coordinate."""
    assert coord in ('x', 'y')
    visual.inserter.insert_vert(
        'gl_Position.{coord} = pos_orig.{coord};'.format(coord=coord),
        'after_transforms')


def subplot_bounds(shape=None, index=None):
    """Get the data bounds of a subplot."""
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
    """Get the data bounds in GLSL of a subplot."""
    x0 = '-1.0 + 2.0 * {i}.y / {s}.y'.format(s=shape, i=index)
    y0 = '+1.0 - 2.0 * ({i}.x + 1) / {s}.x'.format(s=shape, i=index)
    x1 = '-1.0 + 2.0 * ({i}.y + 1) / {s}.y'.format(s=shape, i=index)
    y1 = '+1.0 - 2.0 * ({i}.x) / {s}.x'.format(s=shape, i=index)

    return 'vec4(\n{x0}, \n{y0}, \n{x1}, \n{y1})'.format(x0=x0, y0=y0, x1=x1, y1=y1)


def extend_bounds(bounds_list):
    """Return a single data bounds 4-tuple from a list of data bounds."""
    bounds = np.array(bounds_list)
    xmins, ymins = bounds[:, :2].min(axis=0)
    xmaxs, ymaxs = bounds[:, 2:].max(axis=0)
    xmin, ymin, xmax, ymax = xmins.min(), ymins.min(), xmaxs.max(), ymaxs.max()
    # Avoid degenerate bounds.
    if xmin == xmax:
        xmin, xmax = -1, 1
    if ymin == ymax:
        ymin, ymax = -1, 1
    return (xmin, ymin, xmax, ymax)


def pixels_to_ndc(pos, size=None):
    """Convert from pixels to normalized device coordinates (in [-1, 1])."""
    pos = np.asarray(pos, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    pos = pos / (size / 2.) - 1
    # Flip y, because the origin in pixels is at the top left corner of the
    # window.
    pos[1] = -pos[1]
    return pos


"""Bounds in Normalized Device Coordinates (NDC)."""
NDC = (-1.0, -1.0, +1.0, +1.0)


#------------------------------------------------------------------------------
# Base Transform
#------------------------------------------------------------------------------

class BaseTransform(object):
    """Base class for all transforms."""
    def __init__(self, **kwargs):
        self.__dict__.update(**{k: v for k, v in kwargs.items() if v is not None})

        # Method decorators.
        self.apply = _wrap_apply(self.apply)
        self.glsl = _wrap_glsl(self.glsl)

    def apply(self, arr):
        """Apply the transform to an (n, 2) array."""
        raise NotImplementedError()

    def glsl(self, var):
        """Return the GLSL code for the transform."""
        raise NotImplementedError()

    def inverse(self):
        """Return a Transform instance for the inverse transform."""
        raise NotImplementedError()

    def __add__(self, other):
        return TransformChain().add([self, other])


#------------------------------------------------------------------------------
# Transforms
#------------------------------------------------------------------------------

class Translate(BaseTransform):
    """Translation transform.

    Constructor
    -----------
    amount : 2-tuple
        Coordinates of the translation.
    gpu_var : str
        The name of the GPU variable with the translate vector.

    """

    amount = None
    gpu_var = None

    def __init__(self, amount=None, **kwargs):
        super(Translate, self).__init__(amount=amount, **kwargs)

    def apply(self, arr, param=None):
        """Apply a translation to a NumPy array."""
        assert isinstance(arr, np.ndarray)
        param = param if param is not None else _call_if_callable(self.amount)
        return arr + np.asarray(param)

    def glsl(self, var):
        """Return a GLSL snippet that applies the translation to a given GLSL variable name."""
        assert var
        return '''
        // Translate transform.
        {var} = {var} + {translate};
        '''.format(var=var, translate=self.gpu_var or _call_if_callable(self.amount))

    def inverse(self):
        """Return the inverse Translate instance."""
        return Translate(
            amount=_minus(_call_if_callable(self.amount)) if self.amount is not None else None,
            gpu_var=('-%s' % self.gpu_var) if self.gpu_var else None)


class Scale(BaseTransform):
    """Scale transform.

    Constructor
    -----------
    amount : 2-tuple
        Coordinates of the scaling.
    gpu_var : str
        The name of the GPU variable with the scaling vector.

    """

    amount = None
    gpu_var = None

    def __init__(self, amount=None, **kwargs):
        super(Scale, self).__init__(amount=amount, **kwargs)

    def apply(self, arr, param=None):
        """Apply a scaling to a NumPy array."""
        assert isinstance(arr, np.ndarray)
        param = param if param is not None else _call_if_callable(self.amount)
        return arr * np.asarray(param)

    def glsl(self, var):
        """Return a GLSL snippet that applies the scaling to a given GLSL variable name."""
        assert var
        return '''
        // Translate transform.
        {var} = {var} * {scaling};
        '''.format(var=var, scaling=self.gpu_var or _call_if_callable(self.amount))

    def inverse(self):
        """Return the inverse Scale instance."""
        return Scale(
            amount=_inverse(_call_if_callable(self.amount)) if self.amount is not None else None,
            gpu_var=('1.0 / %s' % self.gpu_var) if self.gpu_var else None)


class Rotate(BaseTransform):
    """Rotation transform, either +90° CW (default) or +90° CCW.

    Constructor
    -----------
    direction : str
        Either `cw` (default) or `ccw`.

    """

    direction = 'cw'

    def __init__(self, direction=None, **kwargs):
        super(Rotate, self).__init__(direction=direction, **kwargs)

    def apply(self, arr, direction=None):
        """Apply a rotation to a NumPy array."""
        assert isinstance(arr, np.ndarray)
        direction = direction or self.direction or 'cw'
        assert direction in ('cw', 'ccw')

        assert arr.ndim == 2
        assert arr.shape[1] == 2
        x, y = arr.T
        if direction == 'ccw':
            return np.c_[-y, x]
        else:
            return np.c_[y, -x]

    def glsl(self, var):
        """Return a GLSL snippet that applies the rotation to a given GLSL variable name."""
        assert var
        direction = self.direction or 'cw'
        assert direction in ('cw', 'ccw')
        m = '' if direction == 'ccw' else '-'
        return '''
        // Rotation transform.
        {var} = {m}vec2(-{var}.y, {var}.x);
        '''.format(var=var, m=m)

    def inverse(self):
        """Return the inverse Rotate instance."""
        direction = self.direction or 'cw'
        assert direction in ('cw', 'ccw')
        return Rotate('cw' if direction == 'ccw' else 'ccw')


class Range(BaseTransform):
    """Linear transform from a source rectangle to a target rectangle.

    Constructor
    -----------

    from_bounds : 4-tuple
        Bounds of the source rectangle.
    to_bounds : 4-tuple
        Bounds of the target rectangle.
    from_gpu_var : str
        Name of the GPU variable with the from bounds.
    to_gpu_var : str
        Name of the GPU variable with the to bounds.

    """

    from_bounds = NDC
    to_bounds = NDC
    from_gpu_var = None
    to_gpu_var = None

    def __init__(self, from_bounds=None, to_bounds=None, **kwargs):
        super(Range, self).__init__(from_bounds=from_bounds, to_bounds=to_bounds, **kwargs)

    def apply(self, arr, from_bounds=None, to_bounds=None):
        """Apply the transform to a NumPy array."""
        from_bounds = from_bounds if from_bounds is not None else self.from_bounds
        to_bounds = to_bounds if to_bounds is not None else self.to_bounds
        assert not isinstance(from_bounds, str) and not isinstance(to_bounds, str)
        from_bounds = np.atleast_2d(_call_if_callable(from_bounds)).astype(np.float64)
        to_bounds = np.atleast_2d(_call_if_callable(to_bounds)).astype(np.float64)
        assert from_bounds.shape[-1] == 4
        assert to_bounds.shape[-1] == 4
        return range_transform(from_bounds, to_bounds, arr)

    def glsl(self, var):
        """Return a GLSL snippet that applies the transform to a given GLSL variable name."""
        assert var

        from_bounds = _glslify(self.from_gpu_var or self.from_bounds)
        to_bounds = _glslify(self.to_gpu_var or self.to_bounds)

        return '''
        // Range transform.
        {var} = ({var} - {f}.xy);
        {var} = {var} * ({t}.zw - {t}.xy);
        {var} = {var} / ({f}.zw - {f}.xy);
        {var} = {var} + {t}.xy;
        '''.format(var=var, f=from_bounds, t=to_bounds)

    def inverse(self):
        """Return the inverse Range instance."""
        return Range(
            from_bounds=self.to_bounds, to_bounds=self.from_bounds,
            from_gpu_var=self.to_gpu_var, to_gpu_var=self.from_gpu_var,
        )


def Subplot(shape=None, index=None, shape_gpu_var=None, index_gpu_var=None):
    """Return a particular Range transform that transforms from NDC to a subplot at a particular
    location, in a grid layout.

    Parameters
    ----------
    shape : 2-tuple
        Number of rows and columns in the grid layout.
    index : 2-tuple
        Index o the row and column of the subplot.
    shape_gpu_var : str
        Name of the GPU variable with the grid's shape.
    index_gpu_var : str
        Name of the GPU variable with the grid's subplot index.

    """
    from_bounds = NDC
    to_bounds = NDC
    from_gpu_var = None
    to_gpu_var = None

    if isinstance(shape, str):
        shape_gpu_var = shape
        shape = None

    if isinstance(index, str):
        index_gpu_var = index
        index = None

    if shape_gpu_var is not None:
        to_gpu_var = subplot_bounds_glsl(shape=shape_gpu_var, index=index_gpu_var)
    if shape is not None:
        if hasattr(shape, '__call__') and hasattr(index, '__call__'):
            to_bounds = lambda: subplot_bounds(shape(), index())
        else:
            to_bounds = subplot_bounds(shape, index)

    return Range(
        from_bounds=from_bounds, to_bounds=to_bounds,
        from_gpu_var=from_gpu_var, to_gpu_var=to_gpu_var)


class Clip(BaseTransform):
    """Transform that discards data outside a given rectangle.

    Constructor
    -----------

    bounds : 4-tuple
        Bounds of the clipping rectangle.

    """

    bounds = NDC

    def __init__(self, bounds=None, **kwargs):
        super(Clip, self).__init__(bounds=bounds, **kwargs)

    def apply(self, arr, bounds=None):
        """Apply the clipping to a NumPy array."""
        bounds = bounds if bounds is not None else _call_if_callable(self.bounds)
        assert isinstance(bounds, (tuple, list))
        assert len(bounds) == 4
        index = ((arr[:, 0] >= bounds[0]) &
                 (arr[:, 1] >= bounds[1]) &
                 (arr[:, 0] <= bounds[2]) &
                 (arr[:, 1] <= bounds[3]))
        return arr[index, ...]

    def glsl(self, var):
        """Return a GLSL snippet that applies the clipping to a given GLSL variable name,
        in the fragment shader."""
        assert var
        bounds = _glslify(self.bounds)

        return """
        // Clip transform.
        if (({var}.x < {bounds}.x) ||
            ({var}.y < {bounds}.y) ||
            ({var}.x > {bounds}.z) ||
            ({var}.y > {bounds}.w)) {{
            discard;
        }}
        """.format(bounds=bounds, var=var)

    def inverse(self):
        """Return the same instance (the inverse has no sense for a Clip transform)."""
        return self


#------------------------------------------------------------------------------
# Transform chain
#------------------------------------------------------------------------------

class TransformChain(object):
    """A linear sequence of transforms."""
    def __init__(self, transforms=None, origin=None):
        self.transformed_var_name = None
        self.origin = origin
        self._transforms = []  # list of tuples (transform, origin)
        if transforms:
            self.add(transforms)

    @property
    def transforms(self):
        """List of transforms."""
        return [t for (t, origin) in self._transforms]

    def add(self, transforms, origin=None):
        """Add some transforms."""
        origin = origin or self.origin
        if not isinstance(transforms, list):
            transforms = [transforms]
        self._transforms.extend([(t, origin) for t in transforms])
        return self

    def get(self, class_name):
        """Get a transform in the chain from its name."""
        for transform, origin in self._transforms:
            if transform.__class__.__name__ == class_name:
                return transform

    def apply(self, arr):
        """Apply all transforms on an array."""
        for t in self.transforms:
            if isinstance(t, Clip):
                continue
            arr = t.apply(arr)
        return arr

    def inverse(self):
        """Return the inverse chain of transforms."""
        inv_transforms = [
            (transform.inverse(), origin)
            for (transform, origin) in self._transforms[::-1]]
        inv = TransformChain()
        inv._transforms = inv_transforms
        return inv

    def __getitem__(self, i):
        return self._transforms[i][0]

    def __add__(self, tc):
        """Concatenate multiple transform chains."""
        if isinstance(tc, BaseTransform):
            return self.add(tc)
        assert isinstance(tc, TransformChain)
        assert tc.transformed_var_name == self.transformed_var_name
        self._transforms.extend(tc._transforms)
        return self
