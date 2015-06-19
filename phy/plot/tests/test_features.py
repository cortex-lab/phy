# -*- coding: utf-8 -*-

"""Test feature plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

import numpy as np

from ..features import FeatureView, plot_features
from ...utils._color import _random_color
from ...io.mock import (artificial_features,
                        artificial_masks,
                        artificial_spike_clusters,
                        artificial_spike_samples)
from ...utils.testing import show_test


# Skip these tests in "make test-quick".
pytestmark = mark.long()


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _test_features(n_spikes=None, n_clusters=None):
    n_channels = 32
    n_features = 3

    features = artificial_features(n_spikes, n_channels, n_features)
    masks = artificial_masks(n_spikes, n_channels)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_samples = artificial_spike_samples(n_spikes).astype(np.float32)

    c = FeatureView()
    c.visual.features = features
    c.background.features = features * 2.
    # Useful to test depth.
    # masks[n_spikes//2:, ...] = 0
    c.visual.masks = masks
    c.add_extra_feature('time', spike_samples)
    matrix = np.empty((2, 2), dtype=object)
    matrix[...] = [[('time', (0, 0)), ((1, 0), (1, 1))],
                   [((2, 1), (1, 0)), ((1, 0), 'time')]]
    c.dimensions_matrix = matrix
    c.visual.spike_clusters = spike_clusters
    c.visual.cluster_colors = np.array([_random_color()
                                        for _ in range(n_clusters)])

    show_test(c)


def test_features_empty():
    _test_features(n_spikes=0, n_clusters=0)


def test_features_full():
    _test_features(n_spikes=100, n_clusters=3)


def test_plot_features():
    n_spikes = 1000
    n_channels = 32
    n_features = 1
    n_clusters = 2

    features = artificial_features(n_spikes, n_channels, n_features)
    masks = artificial_masks(n_spikes, n_channels)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    # Unclustered spikes.
    spike_clusters[::3] = -1
    spike_samples = artificial_spike_samples(n_spikes).astype(np.float32)

    c = plot_features(features[:, :1, :],
                      show=False)
    show_test(c)

    c = plot_features(features,
                      show=False)
    show_test(c)

    c = plot_features(features, show=False)
    show_test(c)

    c = plot_features(features, masks=masks, show=False)
    show_test(c)

    c = plot_features(features, spike_clusters=spike_clusters, show=False)
    show_test(c)

    c = plot_features(features,
                      extra_features={'time': spike_samples},
                      dimensions=['time', (5, 0)],
                      show=False)
    show_test(c)
