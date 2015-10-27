# -*- coding: utf-8 -*-

"""Tests of waveform loader."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises, yield_fixture

from phy.io.mock import artificial_traces, artificial_spike_samples
from phy.utils import Bunch
from ..waveform import (_slice,
                        WaveformLoader,
                        WaveformExtractor,
                        SpikeLoader,
                        )
from ..filter import bandpass_filter, apply_filter


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

@yield_fixture(params=[(None, None), (-1, 2)])
def waveform_loader(request):
    scale_factor, dc_offset = request.param

    n_samples_trace, n_channels = 1000, 5
    h = 10
    n_samples_waveforms = 2 * h
    n_spikes = n_samples_trace // (2 * n_samples_waveforms)

    traces = artificial_traces(n_samples_trace, n_channels)
    spike_samples = artificial_spike_samples(n_spikes,
                                             max_isi=2 * n_samples_waveforms)

    with raises(ValueError):
        WaveformLoader(traces)

    loader = WaveformLoader(traces=traces,
                            n_samples_waveforms=n_samples_waveforms,
                            scale_factor=scale_factor,
                            dc_offset=dc_offset,
                            )
    b = Bunch(loader=loader,
              n_samples_waveforms=n_samples_waveforms,
              n_spikes=n_spikes,
              spike_samples=spike_samples,
              )
    yield b


def test_loader_edge_case():
    wl = WaveformLoader(n_samples_waveforms=3)
    wl.traces = np.random.rand(0, 2)
    wl[0]


def test_loader_simple(waveform_loader):
    b = waveform_loader
    spike_samples = b.spike_samples
    loader = b.loader
    traces = loader.traces
    dc_offset = loader.dc_offset or 0
    scale_factor = loader.scale_factor or 1
    n_samples_traces, n_channels = traces.shape
    n_samples_waveforms = b.n_samples_waveforms
    h = n_samples_waveforms // 2

    assert loader.offset == 0
    assert loader.dc_offset in (dc_offset, None)
    assert loader.scale_factor in (scale_factor, None)

    def _transform(arr):
        return (arr - dc_offset) * scale_factor

    waveforms = loader[spike_samples[10:20]]
    assert waveforms.shape == (10, n_samples_waveforms, n_channels)
    t = spike_samples[15]
    w1 = waveforms[5, ...]
    w2 = _transform(traces[t - h:t + h, :])
    assert np.allclose(w1, w2)

    sl = SpikeLoader(loader, spike_samples)
    assert np.allclose(sl[15], w2)


def test_edges():
    n_samples_trace, n_channels = 100, 10
    n_samples_waveforms = 20

    traces = artificial_traces(n_samples_trace, n_channels)

    # Filter.
    b_filter = bandpass_filter(rate=1000,
                               low=50,
                               high=200,
                               order=3)
    filter_margin = 10

    # Create a loader.
    loader = WaveformLoader(traces,
                            n_samples_waveforms=n_samples_waveforms,
                            filter=lambda x: apply_filter(x, b_filter),
                            filter_margin=filter_margin)

    # Invalid time.
    with raises(ValueError):
        loader._load_at(200000)

    ns = n_samples_waveforms + filter_margin
    assert loader._load_at(0).shape == (ns, n_channels)
    assert loader._load_at(5).shape == (ns, n_channels)
    assert loader._load_at(n_samples_trace - 5).shape == (ns, n_channels)
    assert loader._load_at(n_samples_trace - 1).shape == (ns, n_channels)


def test_loader_channels():
    n_samples_trace, n_channels = 1000, 10
    n_samples_waveforms = 20

    traces = artificial_traces(n_samples_trace, n_channels)

    # Create a loader.
    loader = WaveformLoader(traces, n_samples_waveforms=n_samples_waveforms)
    loader.traces = traces
    channels = [2, 5, 7]
    loader.channels = channels
    assert loader.channels == channels
    assert loader[500].shape == (1, n_samples_waveforms, 3)
    assert loader[[500, 501, 600, 300]].shape == (4, n_samples_waveforms, 3)

    # Test edge effects.
    assert loader[3].shape == (1, n_samples_waveforms, 3)
    assert loader[995].shape == (1, n_samples_waveforms, 3)

    with raises(NotImplementedError):
        loader[500:510]


def test_loader_filter():
    traces = np.c_[np.arange(20), np.arange(20, 40)].astype(np.int32)
    n_samples_trace, n_channels = traces.shape
    h = 3

    def my_filter(x, axis=0):
        return x * x

    loader = WaveformLoader(traces,
                            n_samples_waveforms=(h, h),
                            filter=my_filter,
                            filter_margin=2)

    t = 10
    waveform_filtered = loader[t]
    traces_filtered = my_filter(traces)
    assert np.allclose(waveform_filtered, traces_filtered[t - h:t + h, :])
