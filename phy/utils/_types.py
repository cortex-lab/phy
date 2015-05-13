# -*- coding: utf-8 -*-

"""Utility functions."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext.six import string_types, integer_types


#------------------------------------------------------------------------------
# Various Python utility functions
#------------------------------------------------------------------------------

class Bunch(dict):
    """A dict with additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        return Bunch(super(Bunch, self).copy())


def _is_list(obj):
    return isinstance(obj, list)


def _is_integer(x):
    return isinstance(x, integer_types + (np.generic,))


def _as_int(x):
    if isinstance(x, integer_types):
        return x
    x = np.asscalar(x)
    return x


def _as_list(obj):
    """Ensure an object is a list."""
    if isinstance(obj, string_types):
        return [obj]
    elif not hasattr(obj, '__len__'):
        return [obj]
    else:
        return obj
