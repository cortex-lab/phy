# -*- coding: utf-8 -*-

"""Waveform filtering routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from scipy import signal

from ..ext import six
from ..utils.array import _as_array


#------------------------------------------------------------------------------
# Waveform filtering routines
#------------------------------------------------------------------------------

def bandpass_filter(rate=None, low=None, high=None, order=None):
    """Butterworth bandpass filter."""
    # TODO: implement in a class instead.
    return signal.butter(order,
                         (low/(rate/2.), high/(rate/2.)),
                         'pass')


def apply_filter(x, filter=None):
    x = _as_array(x)
    if x.shape[0] == 0:
        return x
    b, a = filter
    return signal.filtfilt(b, a, x, axis=0)
