# -*- coding: utf-8 -*-

"""Tests of waveform loader."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from phy.io.mock import (artificial_traces,
                         artificial_spike_samples,
                         artificial_masks,
                         )
from ..waveform import (_slice,
                        WaveformLoader,
                        WaveformExtractor,
                        )


#------------------------------------------------------------------------------
# Tests extractor
#------------------------------------------------------------------------------

def test_extract_simple():
    weak = 1.
    strong = 2.
    nc = 4
    ns = 20

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
                           )
    we.set_thresholds(weak=weak, strong=strong)

    # _component()
    comp = we._component(component, n_samples=ns)
    ae(comp.comp_s, [10, 10, 11, 11, 12, 12])
    ae(comp.comp_ch, [0, 1, 0, 1, 0, 1])
    assert (comp.s_min, comp.s_max) == (10 - 3, 12 + 4)

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
    wave_e = we.extract(data, s)
    assert wave_e.shape[1] == wave.shape[1]
    ae(wave[3:6, :2], wave_e[3:6, :2])

    # align()
    wave_a = we.align(wave_e, s)
    assert wave_a.shape == (3 + 5, nc)

    # Test final call.
    s_f, masks_f, wave_f = we(component, data=data, data_t=data)
    assert s_f == s
    ae(masks_f, masks)
    ae(wave_f, wave_a)

    # Tests with a different order.
    we = WaveformExtractor(extract_before=3,
                           extract_after=5,
                           thresholds={'weak': weak,
                                       'strong': strong},
                           )
    s_f_o, masks_f_o, wave_f_o = we(component, data=data, data_t=data)
    assert s_f == s_f_o
    assert np.allclose(wave_f, wave_f_o)
    ae(masks_f_o, [0.5, 1., 0., 0.])


#------------------------------------------------------------------------------
# Tests utility functions
#------------------------------------------------------------------------------

def test_slice():
    assert _slice(0, (20, 20)) == slice(0, 20, None)


#------------------------------------------------------------------------------
# Tests loader
#------------------------------------------------------------------------------

def waveform_loader(do_filter=False, mask_threshold=None):
    n_samples_trace, n_channels = 1000, 5
    h = 10
    n_samples_waveforms = 2 * h
    n_spikes = n_samples_trace // (2 * n_samples_waveforms)
    sample_rate = 2000.

    traces = artificial_traces(n_samples_trace, n_channels)
    spike_samples = artificial_spike_samples(n_spikes,
                                             max_isi=2 * n_samples_waveforms)
    masks = artificial_masks(n_spikes, n_channels)

    loader = WaveformLoader(traces=traces,
                            spike_samples=spike_samples,
                            masks=masks,
                            mask_threshold=mask_threshold,
                            n_samples_waveforms=n_samples_waveforms,
                            filter_order=3 if do_filter else None,
                            sample_rate=sample_rate,
                            )
    return loader


def test_loader_simple():
    loader = waveform_loader()
    spike_samples = loader.spike_samples
    traces = loader.traces
    n_samples_traces, n_channels = traces.shape
    n_samples_waveforms = loader.n_samples_waveforms
    h = n_samples_waveforms // 2

    waveforms = loader[10:20]
    assert waveforms.shape == (10, n_samples_waveforms, n_channels)
    t = spike_samples[15]
    w1 = waveforms[5, ...]
    w2 = traces[t - h:t + h, :]
    assert np.allclose(w1, w2)
    assert np.allclose(loader[15], w2)


def test_edges():
    loader = waveform_loader(do_filter=True)
    ns = loader.n_samples_waveforms + sum(loader._filter_margin)
    nc = loader.n_channels

    assert loader._load_at(0).shape == (ns, nc)
    assert loader._load_at(5).shape == (ns, nc)
    assert loader._load_at(loader.n_samples_trace - 5).shape == (ns, nc)
    assert loader._load_at(loader.n_samples_trace - 1).shape == (ns, nc)


def test_loader_filter_1():
    traces = np.c_[np.arange(20), np.arange(20, 40)].astype(np.int32)
    n_samples_trace, n_channels = traces.shape
    h = 3

    def my_filter(x, axis=0):
        return x * x

    loader = WaveformLoader(spike_samples=np.arange(20),
                            n_samples_waveforms=(h, h),
                            )
    loader.traces = traces
    loader._filter = my_filter

    t = 10
    waveform_filtered = loader[t]
    traces_filtered = my_filter(traces)
    assert np.allclose(waveform_filtered, traces_filtered[t - h:t + h, :])


def test_loader_filter_2():
    loader = waveform_loader(do_filter=True, mask_threshold=.1)
    ns = loader.n_samples_waveforms
    nc = loader.n_channels

    with raises(ValueError):
        loader._load_at(-10)

    assert loader[0].shape == (1, ns, nc)
    assert loader[:].shape == (loader.n_spikes, ns, nc)
