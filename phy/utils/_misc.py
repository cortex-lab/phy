# -*- coding: utf-8 -*-

"""Utility functions."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import sys
from inspect import getargspec

import numpy as np

from ..ext.six import string_types, integer_types
from ..ext.six.moves import builtins


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


def _as_dict(x):
    """Convert a list of tuples to a dict."""
    if isinstance(x, list):
        return dict(x)
    else:
        return x


def _concatenate_dicts(*dicts):
    """Concatenate dictionaries."""
    out = {}
    for dic in dicts:
        out.update(dic)
    return out


def _is_list(obj):
    return isinstance(obj, list)


def _is_integer(x):
    return isinstance(x, integer_types + (np.generic,))


def _as_list(obj):
    """Ensure an object is a list."""
    if isinstance(obj, string_types):
        return [obj]
    elif not hasattr(obj, '__len__'):
        return [obj]
    else:
        return obj


def _fun_arg_count(f):
    """Return the number of arguments of a function.

    WARNING: with methods, only works if the first argument is named 'self'.

    """
    args = getargspec(f).args
    if args and args[0] == 'self':
        args = args[1:]
    return len(args)


def _is_in_ipython():
    return '__IPYTHON__' in dir(builtins)


def _is_interactive():
    """Determine whether the user has requested interactive mode."""
    # The Python interpreter sets sys.flags correctly, so use them!
    if sys.flags.interactive:
        return True

    # IPython does not set sys.flags when -i is specified, so first
    # check it if it is already imported.
    if '__IPYTHON__' not in dir(builtins):
        return False

    # Then we check the application singleton and determine based on
    # a variable it sets.
    try:
        from IPython.config.application import Application as App
        return App.initialized() and App.instance().interact
    except (ImportError, AttributeError):
        return False
