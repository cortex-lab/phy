# -*- coding: utf-8 -*-

"""Tests of spike detection routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from pytest import yield_fixture

from phy.utils.datasets import download_test_data
from phy.electrode import load_probe
from ..spike_detect import (SpikeDetector,
                            )


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def traces():
    path = download_test_data('test-32ch-10s.dat')
    traces = np.fromfile(path, dtype=np.int16).reshape((200000, 32))
    traces = traces[:20000]

    yield traces


@yield_fixture(params=[(True,), (False,)])
def spike_detector(request):
    remap = request.param[0]

    probe = load_probe('1x32_buzsaki')
    channel_mapping = {i: i for i in range(1, 21, 2)} if remap else None

    sd = SpikeDetector()
    sd.use_single_threshold = False
    sample_rate = 20000
    sd.set_metadata(probe,
                    channel_mapping=channel_mapping,
                    sample_rate=sample_rate)

    yield sd


#------------------------------------------------------------------------------
# Test spike detection
#------------------------------------------------------------------------------

def _plot(sd, traces, spike_samples, masks):
    from vispy.app import run
    from phy.plot import plot_traces
    plot_traces(sd.subset_traces(traces),
                spike_samples=spike_samples,
                masks=masks,
                n_samples_per_spike=40)
    run()


def test_detect(spike_detector, traces):
    sd = spike_detector
    spike_samples, masks, _ = sd.detect(traces)

    n_channels = sd.n_channels
    n_spikes = len(spike_samples)

    assert spike_samples.dtype == np.int64
    assert spike_samples.ndim == 1

    assert masks.dtype == np.float32
    assert masks.ndim == 2
    assert masks.shape == (n_spikes, n_channels)

    # _plot(sd, traces, spike_samples, masks)
