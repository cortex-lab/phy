# -*- coding: utf-8 -*-

"""Waveform filtering routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from scipy import signal

from ..utils._types import _as_array


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
    """Apply a filter to an array."""
    x = _as_array(x)
    if x.shape[0] == 0:
        return x
    b, a = filter
    return signal.filtfilt(b, a, x, axis=0)
