
# -*- coding: utf-8 -*-

"""Tests of waveform filtering routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ..filter import bandpass_filter, apply_filter, Filter, Whitening


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_apply_filter():
    """Test bandpass filtering on a combination of two sinusoids."""

    rate = 10000.
    low, high = 100., 200.

    # Create a signal with small and high frequencies.
    t = np.linspace(0., 1., int(rate))
    x = np.sin(2 * np.pi * low / 2 * t) + np.cos(2 * np.pi * high * 2 * t)

    # Filter the signal.
    filter = bandpass_filter(low=low, high=high, order=4, rate=rate)
    ae(apply_filter([], filter=filter), [])

    for x_filtered in (apply_filter(x, filter=filter),
                       Filter(rate=rate, low=low, high=high, order=4)(x),
                       ):

        # Check that the bandpass-filtered signal is weak.
        k = int(2. / low * rate)
        assert np.abs(x[k:-k]).max() >= .9
        assert np.abs(x_filtered[k:-k]).max() <= .1


def test_whitening():
    x = np.random.uniform(size=(100, 10), low=0., high=1.)
    x[:, 1] += .25 * x[:, 0]
    x[:, 5] += .5 * x[:, 0]

    w = Whitening()
    w.fit(x)
    y = w.transform(x)

    assert y.shape == x.shape
