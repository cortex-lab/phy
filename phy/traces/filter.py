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


class Filter(object):
    """Bandpass filter."""
    def __init__(self, rate=None, low=None, high=None, order=None):
        self._filter = bandpass_filter(rate=rate,
                                       low=low,
                                       high=high,
                                       order=order,
                                       )

    def __call__(self, data):
        return apply_filter(data, filter=self._filter)
