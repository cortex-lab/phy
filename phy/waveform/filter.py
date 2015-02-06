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
    return signal.butter(order,
                         (low/(rate/2.), high/(rate/2.)),
                         'pass')


def apply_filter(x, filter=None):
    if x.shape[0] == 0:
        return x
    b, a = filter
    try:
        out_arr = signal.filtfilt(b, a, x, axis=0)
    except TypeError:
        out_arr = np.zeros_like(x)
        for i_ch in range(x.shape[1]):
            out_arr[:, i_ch] = signal.filtfilt(b, a, x[:, i_ch])
    return out_arr
