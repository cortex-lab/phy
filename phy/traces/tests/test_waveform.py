# -*- coding: utf-8 -*-

"""Tests of waveform loader."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
import numpy.random as npr
from pytest import raises

from ...io.mock import artificial_traces
from ..waveform import _slice, WaveformLoader, WaveformExtractor
from ..filter import bandpass_filter, apply_filter


#------------------------------------------------------------------------------
# Tests extractor
#------------------------------------------------------------------------------

def test_extract_simple():
    weak = 1.
    strong = 2.
    nc = 4
    ns = 20
    channels = list(range(nc))
    cpg = {0: channels}
    # graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: []}

    data = np.random.uniform(size=(ns, nc), low=0., high=1.)

    data[10, 0] = 0.5
    data[11, 0] = 1.5
    data[12, 0] = 1.0

    data[10, 1] = 1.5
    data[11, 1] = 2.5
    data[12, 1] = 2.0

    component = np.array([[10, 0],
                          [10, 1],
                          [11, 0],
                          [11, 1],
                          [12, 0],
                          [12, 1],
                          ])

    we = WaveformExtractor(extract_before=3,
                           extract_after=5,
                           thresholds={'weak': weak,
                                       'strong': strong},
                           channels_per_group=cpg,
                           )

    # _component()
    comp = we._component(component, n_samples=ns)
    ae(comp.comp_s, [10, 10, 11, 11, 12, 12])
    ae(comp.comp_ch, [0, 1, 0, 1, 0, 1])
    assert (comp.s_min, comp.s_max) == (10 - 3, 12 + 4)
    ae(comp.channels, range(nc))

    # _normalize()
    assert we._normalize(weak) == 0
    assert we._normalize(strong) == 1
    ae(we._normalize([(weak + strong) / 2.]), [.5])

    # _comp_wave()
    wave = we._comp_wave(data, comp)
    assert wave.shape == (3 + 5 + 1, nc)
    ae(wave[3:6, :], [[0.5, 1.5, 0., 0.],
                      [1.5, 2.5, 0., 0.],
                      [1.0, 2.0, 0., 0.]])

    # masks()
    masks = we.masks(data, wave, comp)
    ae(masks, [.5, 1., 0, 0])

    # spike_sample_aligned()
    s = we.spike_sample_aligned(wave, comp)
    assert 11 <= s < 12

    # extract()
    wave_e = we.extract(data, s, channels=channels)
    assert wave_e.shape[1] == wave.shape[1]
    ae(wave[3:6, :2], wave_e[3:6, :2])

    # align()
    wave_a = we.align(wave_e, s)
    assert wave_a.shape == (3 + 5, nc)

    # Test final call.
    groups, s_f, wave_f, masks_f = we(component, data=data, data_t=data)
    assert s_f == s
    assert np.all(groups == 0)
    ae(masks_f, masks)
    ae(wave_f, wave_a)

    # Tests with a different order.
    we = WaveformExtractor(extract_before=3,
                           extract_after=5,
                           thresholds={'weak': weak,
                                       'strong': strong},
                           channels_per_group={0: [1, 0, 3]},
                           )
    groups, s_f_o, wave_f_o, masks_f_o = we(component, data=data, data_t=data)
    assert np.all(groups == 0)
    assert s_f == s_f_o
    assert np.allclose(wave_f[:, [1, 0, 3]], wave_f_o)
    ae(masks_f_o, [1., 0.5, 0.])


#------------------------------------------------------------------------------
# Tests loader
#------------------------------------------------------------------------------

def test_slice():
    assert _slice(0, (20, 20)) == slice(0, 20, None)


def test_loader():
    n_samples_trace, n_channels = 10000, 100
    n_samples = 40
    n_spikes = n_samples_trace // (2 * n_samples)

    traces = artificial_traces(n_samples_trace, n_channels)
    spike_samples = np.cumsum(npr.randint(low=0, high=2 * n_samples,
                                          size=n_spikes))

    with raises(ValueError):
        WaveformLoader(traces)

    # Create a loader.
    loader = WaveformLoader(traces, n_samples=n_samples)
    assert id(loader.traces) == id(traces)
    loader.traces = traces

    # Extract a waveform.
    t = spike_samples[10]
    waveform = loader._load_at(t)
    assert waveform.shape == (n_samples, n_channels)
    ae(waveform, traces[t - 20:t + 20, :])

    waveforms = loader[spike_samples[10:20]]
    assert waveforms.shape == (10, n_samples, n_channels)
    t = spike_samples[15]
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
    filter_margin = 10

    # Create a loader.
    loader = WaveformLoader(traces,
                            n_samples=n_samples,
                            filter=lambda x: apply_filter(x, b_filter),
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
    spike_samples = np.cumsum(npr.randint(low=0, high=2 * n_samples,
                                          size=n_spikes))

    # With filter.
    def my_filter(x):
        return x * x

    loader = WaveformLoader(traces,
                            n_samples=(n_samples // 2, n_samples // 2),
                            filter=my_filter,
                            filter_margin=5)

    t = spike_samples[5]
    waveform_filtered = loader._load_at(t)
    traces_filtered = my_filter(traces)
    traces_filtered[t - 20:t + 20, :]
    assert np.allclose(waveform_filtered, traces_filtered[t - 20:t + 20, :])
