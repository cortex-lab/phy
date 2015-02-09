
# -*- coding: utf-8 -*-

"""Tests of waveform loader."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
import numpy.random as npr
from pytest import raises

from ...io.mock.artificial import artificial_traces
from ..loader import _slice, WaveformLoader
from ..filter import bandpass_filter, apply_filter


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_slice():
    assert _slice(0, (20, 20)) == slice(0, 20, None)


def test_loader():
    n_samples_trace, n_channels = 10000, 100
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
    waveform = loader._load_at(t)
    assert waveform.shape == (n_samples, n_channels)
    ae(waveform, traces[t - 20:t + 20, :])

    waveforms = loader[spike_times[10:20]]
    assert waveforms.shape == (10, n_samples, n_channels)
    t = spike_times[15]
    w1 = waveforms[5, ...]
    w2 = traces[t - 20:t + 20, :]
    assert np.allclose(w1, w2)


def test_edges():
    n_samples_trace, n_channels = 1000, 10
    n_samples = 40

    traces = artificial_traces(n_samples_trace, n_channels)

    # Filter.
    b_filter = bandpass_filter(rate=1000,
                               low=50,
                               high=200,
                               order=3)
    filter = lambda x: apply_filter(x, b_filter)
    filter_margin = 10

    # Create a loader.
    loader = WaveformLoader(traces,
                            n_samples=n_samples,
                            filter=filter,
                            filter_margin=filter_margin)

    # Invalid time.
    with raises(ValueError):
        loader._load_at(200000)

    assert loader._load_at(0).shape == (n_samples, n_channels)
    assert loader._load_at(5).shape == (n_samples, n_channels)
    assert loader._load_at(n_samples_trace-5).shape == (n_samples, n_channels)
    assert loader._load_at(n_samples_trace-1).shape == (n_samples, n_channels)


def test_loader_channels():
    n_samples_trace, n_channels = 1000, 50
    n_samples = 40

    traces = artificial_traces(n_samples_trace, n_channels)

    # Create a loader.
    loader = WaveformLoader(traces, n_samples=n_samples)
    loader.traces = traces
    channels = [10, 20, 30]
    loader.channels = channels
    assert loader.channels == channels
    assert loader[500].shape == (1, n_samples, 3)
    assert loader[[500, 501, 600, 300]].shape == (4, n_samples, 3)

    # Test edge effects.
    assert loader[3].shape == (1, n_samples, 3)
    assert loader[995].shape == (1, n_samples, 3)

    with raises(NotImplementedError):
        loader[500:510]


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
    waveform_filtered = loader._load_at(t)
    traces_filtered = my_filter(traces)
    traces_filtered[t - 20:t + 20, :]
    assert np.allclose(waveform_filtered, traces_filtered[t - 20:t + 20, :])
