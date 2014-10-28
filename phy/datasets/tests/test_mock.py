# -*- coding: utf-8 -*-

"""Tests of mock datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..mock import artificial_waveforms, artificial_traces


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _plot(arr):
    import matplotlib.pyplot as plt
    plt.plot(arr)
    plt.show()


def test_artificial():
    nspikes = 10
    nsamples_waveforms = 32
    nsamples_traces = 1000
    nchannels = 64

    # Waveforms.
    waveforms = artificial_waveforms(nspikes=nspikes,
                                     nsamples=nsamples_waveforms,
                                     nchannels=nchannels)
    assert waveforms.shape == (nspikes, nsamples_waveforms, nchannels)

    # Traces.
    traces = artificial_traces(nsamples=nsamples_traces,
                               nchannels=nchannels)
    assert traces.shape == (nsamples_traces, nchannels)
