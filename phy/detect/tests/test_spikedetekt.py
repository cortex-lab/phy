# -*- coding: utf-8 -*-

"""Tests of clustering algorithms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_equal as ae
from pytest import mark

from ...utils.logging import set_level
from ...utils.testing import show_test
from ..spikedetekt import (SpikeDetekt, _split_spikes, _concat, _concatenate)


#------------------------------------------------------------------------------
# Tests spike detection
#------------------------------------------------------------------------------

def setup():
    set_level('info')


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


def test_spike_detect_methods(tempdir, raw_dataset):
    params = raw_dataset.params
    probe = raw_dataset.probe
    sample_rate = raw_dataset.sample_rate
    sd = SpikeDetekt(tempdir=tempdir,
                     probe=raw_dataset.probe,
                     sample_rate=sample_rate,
                     **params)
    traces = raw_dataset.traces
    n_samples = raw_dataset.n_samples
    n_channels = raw_dataset.n_channels

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

    if not split:
        return
    samples = _concat(split[0]['spike_samples'], np.float64)
    waveforms = _concat(split[0]['waveforms'], np.float32)
    masks = _concat(split[0]['masks'], np.float32)

    n_spikes = len(samples)
    n_channels = len(probe['channel_groups'][0]['channels'])

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
def test_spike_detect_real_data(tempdir, raw_dataset):

    params = raw_dataset.params
    probe = raw_dataset.probe
    sample_rate = raw_dataset.sample_rate
    sd = SpikeDetekt(tempdir=tempdir,
                     probe=probe,
                     sample_rate=sample_rate,
                     **params)
    traces = raw_dataset.traces
    n_samples = raw_dataset.n_samples
    npc = params['n_features_per_channel']
    n_samples_w = params['extract_s_before'] + params['extract_s_after']

    # Run the detection.
    out = sd.run_serial(traces, interval_samples=(0, n_samples))

    channels = probe['channel_groups'][0]['channels']
    n_channels = len(channels)

    spike_samples = _concatenate(out.spike_samples[0])
    masks = _concatenate(out.masks[0])
    features = _concatenate(out.features[0])
    n_spikes = out.n_spikes_per_group[0]

    if n_spikes:
        assert spike_samples.shape == (n_spikes,)
        assert masks.shape == (n_spikes, n_channels)
        assert features.shape == (n_spikes, n_channels, npc)

        # There should not be any spike with only masked channels.
        assert np.all(masks.max(axis=1) > 0)

    # Plot...
    from phy.plot.traces import plot_traces
    c = plot_traces(traces[:30000, channels],
                    spike_samples=spike_samples,
                    masks=masks,
                    n_samples_per_spike=n_samples_w,
                    show=False)
    show_test(c)
