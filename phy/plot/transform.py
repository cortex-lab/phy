# -*- coding: utf-8 -*-

"""Transforms."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from textwrap import dedent

import numpy as np

import logging

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Transforms
#------------------------------------------------------------------------------

def _wrap_apply(f):
    def wrapped(arr):
        if arr is None or not len(arr):
            return arr
        arr = np.atleast_2d(arr)
        arr = arr.astype(np.float32)
        assert arr.ndim == 2
        assert arr.shape[1] == 2
        out = f(arr)
        out = out.astype(np.float32)
        assert out.ndim == 2
        assert out.shape[0] == arr.shape[0]
        return out
    return wrapped


class BaseTransform(object):
    def __init__(self):
        self.apply = _wrap_apply(self.apply)

    def apply(self, arr):
        raise NotImplementedError()


class Translate(BaseTransform):
    def __init__(self, txy):
        BaseTransform.__init__(self)
        self.txy = np.asarray(txy)

    def apply(self, arr):
        return arr + self.txy

    def glsl(self, var):
        return """{} + {}""".format(var, self.txy)


class Scale(BaseTransform):
    def __init__(self, sxy):
        BaseTransform.__init__(self)
        self.sxy = np.asarray(sxy)

    def apply(self, arr):
        return arr * self.sxy

    def glsl(self, var):
        return """{} * {}""".format(var, self.sxy)


class Range(BaseTransform):
    def __init__(self, from_range, to_range):
        BaseTransform.__init__(self)

        self.f0 = np.array(from_range[:2])
        self.f1 = np.array(from_range[2:])
        self.t0 = np.array(to_range[:2])
        self.t1 = np.array(to_range[2:])

    def apply(self, arr):
        f0, f1, t0, t1 = self.f0, self.f1, self.t0, self.t1
        return t0 + (t1 - t0) * (arr - f0) / (f1 - f0)

    def glsl(self, var):
        return dedent("""
            {t0} + ({t1} - {t0}) * ({var} - {f0}) / ({f1} - {f0})
        """).format(f0=self.f0, f1=self.f1, t0=self.t0, t1=self.t1).strip()


class Clip(BaseTransform):
    def __init__(self, bounds):
        BaseTransform.__init__(self)
        self.xymin = np.asarray(bounds[0:2])
        self.xymax = np.asarray(bounds[2:])

    def apply(self, arr):
        return np.clip(arr, self.xymin, self.xymax)

    def glsl(self, var):
        return dedent("""
            if (({var}.x < {xymin}.x) |
                ({var}.y < {xymin}.y) |
                ({var}.x > {xymax}.x) |
                ({var}.y > {xymax}.y)) {
                discard;
            }
        """).format(xymin=self.xymin,
                    xymax=self.xymax,
                    ).strip()
