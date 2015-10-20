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
        assert out.shape == arr.shape
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
    def __init__(self, xymin, xymax, mode='hard'):
        BaseTransform.__init__(self)
        self.xymin = np.asarray(xymin)
        self.xymax = np.asarray(xymax)

        # Only if the variables are numbers, not strings.
        if not isinstance(xymin, string_types):
            self.xymax_minus_xymin = self.xymax - self.xymin

        self.mode = mode

    def apply(self, arr):
        if self.mode == 'hard':
            xym = arr.min(axis=0)
            xyM = arr.max(axis=0)

            # Handle min=max degenerate cases.
            for i in range(arr.shape[1]):
                if np.allclose(xym[i], xyM[i]):
                    arr[:, i] += .5
                    xyM[i] += 1

            return self.xymin + self.xymax_minus_xymin * \
                (arr - xym) / (xyM - xym)

        raise NotImplementedError()

    def glsl(self, var):
        return TODO


class Clip(BaseTransform):
    def __init__(self, xymin, xymax):
        BaseTransform.__init__(self)
        self.xymin = np.asarray(xymin)
        self.xymax = np.asarray(xymax)

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
                    )


class GPU(BaseTransform):
    pass
