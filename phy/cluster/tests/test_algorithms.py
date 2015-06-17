# -*- coding: utf-8 -*-

"""Tests of clustering algorithms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np

from ...utils.tempdir import TemporaryDirectory
from ...utils.settings import Settings
from ..algorithms import cluster, SpikeDetekt
from ...io.kwik import KwikModel
from ...io.kwik.mock import create_mock_kwik
from ...io.mock import artificial_traces


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_spike_detect():
    sample_rate = 10000
    n_samples = 10000
    n_channels = 4
    traces = artificial_traces(n_samples, n_channels)

    # Load default settings.
    curdir = op.dirname(op.realpath(__file__))
    default_settings_path = op.join(curdir, '../default_settings.py')
    settings = Settings(default_path=default_settings_path)
    params = settings['spikedetekt_params'](sample_rate)
    params['sample_rate'] = sample_rate

    sd = SpikeDetekt(**params)

    # Filter the data.
    traces_f = sd.apply_filter(traces)
    assert traces_f.shape == traces.shape
    assert not np.any(np.isnan(traces_f))

    # Thresholds.
    weak, strong = sd.find_thresholds(traces)
    assert 0 < weak < strong


def test_cluster():
    n_spikes = 100
    with TemporaryDirectory() as tempdir:
        filename = create_mock_kwik(tempdir,
                                    n_clusters=1,
                                    n_spikes=n_spikes,
                                    n_channels=8,
                                    n_features_per_channel=3,
                                    n_samples_traces=5000)
        model = KwikModel(filename)

        spike_clusters = cluster(model, num_starting_clusters=10)
        assert len(spike_clusters) == n_spikes

        spike_clusters = cluster(model, num_starting_clusters=10,
                                 spike_ids=range(100))
        assert len(spike_clusters) == 100
