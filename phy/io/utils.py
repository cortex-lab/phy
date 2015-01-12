# -*- coding: utf-8 -*-

"""IO-related utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from math import floor
import numpy as np

from ..ext import six


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------
def _range_from_slice(myslice, start=None, stop=None, length=None):
    """Convert a slice to an array of integers."""
    assert isinstance(myslice, slice)
    # Find 'step'.
    step = myslice.step if myslice.step is not None else 1
    # Find 'start'.
    start = start if start is not None else myslice.start
    if start is None:
        start = 0
    # Find 'stop' as a function of length if 'stop' is unspecified.
    if length is not None:
        stop = floor(start + step * length)
    else:
        stop = stop if stop is not None else myslice.stop
    myrange = np.arange(start, stop, step)
    # Check the length if it was specified.
    if length is not None:
        assert len(myrange) == length
    return myrange
