
# -*- coding: utf-8 -*-

"""Tests of waveform loader."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal
import numpy.random as npr
from pytest import raises

from ...datasets.mock import artificial_traces
from ..loader import _slice, WaveformLoader


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_slice():
    assert _slice(0, (20, 20)) == slice(0, 20, None)


def test_loader():
    n_samples_trace, n_channels = 1000, 100
    n_samples = 40
    n_spikes = n_samples_trace // (2 * n_samples)

    traces = artificial_traces(n_samples_trace, n_channels)
    spike_times = np.cumsum(npr.randint(low=0, high=2 * n_samples,
                                        size=n_spikes))

    with raises(ValueError):
        WaveformLoader(traces)

    # Create a loader.
    loader = WaveformLoader(traces, n_samples=n_samples)
    assert id(loader.traces) == id(traces)
    loader.traces = traces

    # Extract a waveform.
    t = spike_times[10]
    waveform = loader.load_at(t)
    assert waveform.shape == (n_samples, n_channels)
    assert_array_equal(waveform, traces[t - 20:t + 20, :])

    # Invalid time.
    with raises(ValueError):
        loader.load_at(2000)


def test_loader_filter():
    n_samples_trace, n_channels = 1000, 100
    n_samples = 40
    n_spikes = n_samples_trace // (2 * n_samples)

    traces = artificial_traces(n_samples_trace, n_channels)
    spike_times = np.cumsum(npr.randint(low=0, high=2 * n_samples,
                                        size=n_spikes))

    # With filter.
    def my_filter(x):
        return x * x

    loader = WaveformLoader(traces,
                            n_samples=(n_samples // 2, n_samples // 2),
                            filter=my_filter,
                            filter_margin=5)

    t = spike_times[5]
    waveform_filtered = loader.load_at(t)
    traces_filtered = my_filter(traces)
    traces_filtered[t - 20:t + 20, :]
    assert np.allclose(waveform_filtered, traces_filtered[t - 20:t + 20, :])
