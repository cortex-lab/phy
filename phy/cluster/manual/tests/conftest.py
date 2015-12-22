# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from pytest import yield_fixture

from phy.electrode.mea import staggered_positions
from phy.io.array import _spikes_per_cluster
from phy.io.mock import (artificial_waveforms,
                         artificial_features,
                         artificial_spike_clusters,
                         artificial_spike_samples,
                         artificial_masks,
                         artificial_traces,
                         )
from phy.utils import Bunch
from phy.cluster.manual.store import get_closest_clusters


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def cluster_ids():
    yield [0, 1, 2, 10, 11, 20, 30]
    #      i, g, N,  i,  g,  N, N


@yield_fixture
def cluster_groups():
    yield {0: 'noise', 1: 'good', 10: 'mua', 11: 'good'}


@yield_fixture
def quality():
    yield lambda c: c


@yield_fixture
def similarity(cluster_ids):
    sim = lambda c, d: (c * 1.01 + d)
    yield lambda c: get_closest_clusters(c, cluster_ids, sim)


@yield_fixture
def model(tempdir):
    model = Bunch()

    n_spikes = 51
    n_samples_w = 31
    n_samples_t = 20000
    n_channels = 11
    n_clusters = 3
    n_features = 4

    model.path = op.join(tempdir, 'test')
    model.n_channels = n_channels
    # TODO: test with permutation and dead channels
    model.channel_order = None
    model.n_spikes = n_spikes
    model.sample_rate = 20000.
    model.duration = n_samples_t / float(model.sample_rate)
    model.spike_times = artificial_spike_samples(n_spikes) * 1.
    model.spike_times /= model.spike_times[-1]
    model.spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    model.cluster_ids = np.unique(model.spike_clusters)
    model.channel_positions = staggered_positions(n_channels)
    model.waveforms = artificial_waveforms(n_spikes, n_samples_w, n_channels)
    model.masks = artificial_masks(n_spikes, n_channels)
    model.traces = artificial_traces(n_samples_t, n_channels)
    model.features = artificial_features(n_spikes, n_channels, n_features)

    # features_masks array
    f = model.features.reshape((n_spikes, -1))
    m = np.repeat(model.masks, n_features, axis=1)
    model.features_masks = np.dstack((f, m))

    model.spikes_per_cluster = _spikes_per_cluster(model.spike_clusters)
    model.n_features_per_channel = n_features
    model.n_samples_waveforms = n_samples_w
    model.cluster_groups = {c: None for c in range(n_clusters)}

    yield model
