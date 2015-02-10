# -*- coding: utf-8 -*-

"""Utility functions."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from inspect import getargspec


#------------------------------------------------------------------------------
# Various Python utility functions
#------------------------------------------------------------------------------

def _as_dict(x):
    """Convert a list of tuples to a dict."""
    if isinstance(x, list):
        return dict(x)
    else:
        return x


def _fun_arg_count(f):
    """Return the number of arguments of a function.

    WARNING: with methods, only works if the first argument is named 'self'.

    """
    args = getargspec(f).args
    if args and args[0] == 'self':
        args = args[1:]
    return len(args)
