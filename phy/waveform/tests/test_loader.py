
# -*- coding: utf-8 -*-

"""Tests of waveform loader."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
import numpy.random as npr
from pytest import raises

from ...datasets.mock import artificial_traces
from ..loader import WaveformLoader


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_loader():
    n_samples_trace, n_channels = 1000, 100
    n_samples = 40
    n_spikes = n_samples_trace // (2 * n_samples)

    traces = artificial_traces(n_samples_trace, n_channels)
    spike_times = np.cumsum(npr.randint(low=0, high=2 * n_samples,
                                        size=n_spikes))

    loader = WaveformLoader(traces, spike_times=spike_times,
                            n_samples=n_samples)
    assert loader
