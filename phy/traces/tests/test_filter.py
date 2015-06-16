
# -*- coding: utf-8 -*-

"""Tests of waveform filtering routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ..filter import bandpass_filter, apply_filter


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_apply_filter():
    """Test bandpass filtering on a combination of two sinusoids."""

    rate = 10000.
    low, high = 100., 200.

    # Create a signal with small and high frequencies.
    t = np.linspace(0., 1., rate)
    x = np.sin(2 * np.pi * low / 2 * t) + np.cos(2 * np.pi * high * 2 * t)

    # Filter the signal.
    filter = bandpass_filter(low=low, high=high, order=4, rate=rate)
    x_filtered = apply_filter(x, filter=filter)

    # Check that the bandpass-filtered signal is weak.
    k = int(2. / low * rate)
    assert np.abs(x[k:-k]).max() >= .9
    assert np.abs(x_filtered[k:-k]).max() <= .1

    ae(apply_filter([], filter=filter), [])
