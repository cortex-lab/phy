# -*- coding: utf-8 -*-

"""Utility functions."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from six import string_types, integer_types


#------------------------------------------------------------------------------
# Various Python utility functions
#------------------------------------------------------------------------------

_ACCEPTED_ARRAY_DTYPES = (np.float, np.float32, np.float64,
                          np.int, np.int8, np.int16, np.uint8, np.uint16,
                          np.int32, np.int64, np.uint32, np.uint64,
                          np.bool)


class Bunch(dict):
    """A dict with additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        return Bunch(super(Bunch, self).copy())


def _bunchify(b):
    """Ensure all dict elements are Bunch."""
    assert isinstance(b, dict)
    b = Bunch(b)
    for k in b:
        if isinstance(b[k], dict):
            b[k] = Bunch(b[k])
    return b


def _is_list(obj):
    return isinstance(obj, list)


def _as_scalar(obj):
    if isinstance(obj, np.generic):
        return np.asscalar(obj)
    assert isinstance(obj, (int, float))
    return obj


def _as_scalars(arr):
    return [_as_scalar(x) for x in arr]


def _is_integer(x):
    return isinstance(x, integer_types + (np.generic,))


def _is_float(x):
    return isinstance(x, (float, np.float32, np.float64))


def _as_list(obj):
    """Ensure an object is a list."""
    if obj is None:
        return None
    elif isinstance(obj, string_types):
        return [obj]
    elif isinstance(obj, tuple):
        return list(obj)
    elif not hasattr(obj, '__len__'):
        return [obj]
    else:
        return obj


def _is_array_like(arr):
    return isinstance(arr, (list, np.ndarray))


def _as_array(arr, dtype=None):
    """Convert an object to a numerical NumPy array.

    Avoid a copy if possible.

    """
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and dtype is None:
        return arr
    if isinstance(arr, integer_types + (float,)):
        arr = [arr]
    out = np.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError("'arr' seems to have an invalid dtype: "
                         "{0:s}".format(str(out.dtype)))
    return out


def _as_tuple(item):
    """Ensure an item is a tuple."""
    if item is None:
        return None
    elif not isinstance(item, tuple):
        return (item,)
    else:
        return item
