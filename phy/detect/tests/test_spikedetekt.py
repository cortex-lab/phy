# -*- coding: utf-8 -*-

"""Tests of clustering algorithms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_equal as ae
from pytest import fixture, mark

from ...utils.datasets import _download_test_data
from ...utils.logging import set_level
from ...utils.settings import _load_default_settings
from ...utils.testing import show_test
from ...electrode.mea import load_probe
from ...io.mock import artificial_traces
from ..spikedetekt import (SpikeDetekt, _split_spikes,
                           _concat, SpikeCounts)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def setup():
    set_level('info')


def teardown():
    set_level('info')


sample_rate = 10000
n_samples = 25000
n_channels = 4


def _spikedetekt(tempdir, n_groups=2, type=None):
    if type == 'artificial':
        traces = artificial_traces(n_samples, n_channels)
    elif type == 'null':
        traces = np.zeros((n_samples, n_channels))
    traces[5000:5010, 1] *= 5
    traces[15000:15010, 3] *= 5

    # Load default settings.
    settings = _load_default_settings()
    params = settings['spikedetekt']
    params['sample_rate'] = sample_rate
    params['use_single_threshold'] = False

    if n_groups == 1:
        params['probe_adjacency_list'] = {0: [1, 2],
                                          1: [0, 2],
                                          2: [0, 1],
                                          3: []}
        params['probe_channels'] = {0: [0, 1, 2, 3]}
    elif n_groups == 2:
        params['probe_adjacency_list'] = {0: [1, 2],
                                          1: [0, 2],
                                          2: [0, 1],
                                          3: []}
        params['probe_channels'] = {0: [0, 1, 2], 1: [3]}

    sd = SpikeDetekt(tempdir=tempdir, **params)

    return sd, traces, params


@fixture(params=['artificial', 'null'])
def spikedetekt(request, tempdir):
    return _spikedetekt(tempdir, type=request.param)


@fixture(params=['artificial', 'null'])
def spikedetekt_one_group(request, tempdir):
    return _spikedetekt(tempdir, n_groups=1, type=request.param)


#------------------------------------------------------------------------------
# Tests spike detection
#------------------------------------------------------------------------------

def test_spike_counts():
    c = {0: {10: 100, 20: 200},
         2: {10: 1, 30: 300},
         }
    sc = SpikeCounts(c)
    assert sc() == 601

    assert sc(group=0) == 300
    assert sc(group=1) == 0
    assert sc(group=2) == 301

    assert sc(chunk=10) == 101
    assert sc(chunk=20) == 200
    assert sc(chunk=30) == 300


def test_split_spikes():
    groups = np.zeros(10, dtype=np.int)
    groups[1::2] = 1

    idx = np.ones(10, dtype=np.bool)
    idx[0] = False
    idx[-1] = False

    a = np.random.rand(10, 2)
    b = np.random.rand(10, 3, 2)

    out = _split_spikes(groups, idx, a=a, b=b)

    assert sorted(out) == [0, 1]
    assert sorted(out[0]) == ['a', 'b']
    assert sorted(out[1]) == ['a', 'b']

    ae(out[0]['a'], a[1:-1][1::2])
    ae(out[0]['b'], b[1:-1][1::2])

    ae(out[1]['a'], a[1:-1][::2])
    ae(out[1]['b'], b[1:-1][::2])


def test_spike_detect_methods(spikedetekt_one_group):
    sd, traces, params = spikedetekt_one_group

    # Filter the data.
    traces_f = sd.apply_filter(traces)
    assert traces_f.shape == traces.shape
    assert not np.any(np.isnan(traces_f))

    # Thresholds.
    thresholds = sd.find_thresholds(traces)
    assert np.all(0 <= thresholds['weak'])
    assert np.all(thresholds['weak'] <= thresholds['strong'])

    # Spike detection.
    traces_f[1000:1010, :3] *= 5
    traces_f[2000:2010, [0, 2]] *= 5
    traces_f[3000:3020, :] *= 5
    components = sd.detect(traces_f, thresholds)
    assert isinstance(components, list)
    # n_spikes = len(components)
    n_samples_waveforms = (params['extract_s_before'] +
                           params['extract_s_after'])

    # Spike extraction.
    split = sd.extract_spikes(components, traces_f, thresholds,
                              keep_bounds=(0, n_samples))

    samples = _concat(split[0]['spike_samples'], np.float64)
    waveforms = _concat(split[0]['waveforms'], np.float32)
    masks = _concat(split[0]['masks'], np.float32)

    n_spikes = len(samples)

    assert samples.dtype == np.float64
    assert samples.shape == (n_spikes,)
    assert waveforms.shape == (n_spikes, n_samples_waveforms, n_channels)
    assert masks.shape == (n_spikes, n_channels)
    assert 0. <= masks.min() < masks.max() <= 1.
    assert not np.any(np.isnan(samples))
    assert not np.any(np.isnan(waveforms))
    assert not np.any(np.isnan(masks))

    # PCA.
    pcs = sd.waveform_pcs(waveforms, masks)
    n_pcs = params['n_features_per_channel']
    assert pcs.shape == (n_pcs, n_samples_waveforms, n_channels)
    assert not np.any(np.isnan(pcs))

    # Features.
    features = sd.features(waveforms, pcs)
    assert features.shape == (n_spikes, n_channels, n_pcs)
    assert not np.any(np.isnan(features))


@mark.long
def test_spike_detect_real_data(tempdir, spikedetekt):
    # Set the parameters.
    settings = _load_default_settings()
    sample_rate = 20000
    params = settings['spikedetekt']
    params['sample_rate'] = sample_rate

    n_channels = 32
    npc = params['n_features_per_channel']
    n_samples_w = params['extract_s_before'] + params['extract_s_after']
    probe = load_probe('1x32_buzsaki')

    # Load the traces.
    path = _download_test_data('test-32ch-10s.dat')
    traces = np.fromfile(path, dtype=np.int16).reshape((200000, 32))

    # Run the detection.
    sd = SpikeDetekt(tempdir=tempdir, probe=probe, **params)
    out = sd.run_serial(traces, interval_samples=(0, 50000))

    n_spikes = out.n_spikes_total

    def _concat(arrs):
        return np.concatenate(list(arrs))

    spike_samples = _concat(out.spike_samples[0])
    masks = _concat(out.masks[0])
    features = _concat(out.features[0])

    assert spike_samples.shape == (n_spikes,)
    assert masks.shape == (n_spikes, n_channels)
    assert features.shape == (n_spikes, n_channels, npc)

    # There should not be any spike with only masked channels.
    assert np.all(masks.max(axis=1) > 0)

    # Plot...
    from phy.plot.traces import plot_traces
    c = plot_traces(traces[:30000, :],
                    spike_samples=spike_samples,
                    masks=masks,
                    n_samples_per_spike=n_samples_w,
                    show=False)
    show_test(c)
