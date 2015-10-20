# -*- coding: utf-8 -*-

"""Transforms."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

import logging

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Transforms
#------------------------------------------------------------------------------

def _wrap_apply(f):
    def wrapped(arr):
        arr = np.atleast_2d(arr)
        assert arr.ndim == 2
        assert arr.shape[1] == 2
        out = f(arr)
        assert out.shape == arr.shape
        return out
    return wrapped


class BaseTransform(object):
    def __init__(self):
        self.apply = _wrap_apply(self.apply)

    def apply(self, arr):
        raise NotImplementedError()


class Translate(BaseTransform):
    def __init__(self, tx, ty):
        BaseTransform.__init__(self)
        self.tx, self.ty = tx, ty

    def apply(self, arr):
        return arr + np.array([[self.tx, self.ty]])


class Scale(BaseTransform):
    def __init__(self, sx, sy):
        BaseTransform.__init__(self)
        self.sx, self.sy = sx, sy

    def apply(self, arr):
        return arr * np.array([[self.sx, self.sy]])


class Range(BaseTransform):
    def __init__(self, xmin, ymin, xmax, ymax, mode='hard'):
        BaseTransform.__init__(self)
        self.xmin, ymin = xmin, ymin
        self.xmax, ymax = xmax, ymax
        self.mode = mode

    def apply(self, arr):
        if self.mode == 'hard':
            xym = arr.min(axis=0)
            xyM = arr.max(axis=0)

            xymin = np.array([[self.xmin, self.ymin]])
            xymax = np.array([[self.xmax, self.ymax]])

            xymin + (xymax - xymin) * (arr - xym) / (xyM - xym)

        raise NotImplementedError()


class Clip(BaseTransform):
    def __init__(self, xmin, ymin, xmax, ymax):
        BaseTransform.__init__(self)
        self.xmin, ymin = xmin, ymin
        self.xmax, ymax = xmax, ymax

    def apply(self, arr):
        pass


class GPU(BaseTransform):
    pass