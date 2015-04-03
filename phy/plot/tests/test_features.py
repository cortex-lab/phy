# -*- coding: utf-8 -*-

"""Test feature plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.logging import set_level
from ..features import FeatureView
from ...utils._color import _random_color
from ...io.mock.artificial import (artificial_features,
                                   artificial_masks,
                                   artificial_spike_clusters,
                                   artificial_spike_times)
from ...utils.testing import show_test


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def teardown():
    pass


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def _test_features(n_spikes=None, n_clusters=None):
    n_channels = 32
    n_features = 3

    features = artificial_features(n_spikes, n_channels, n_features)
    masks = artificial_masks(n_spikes, n_channels)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_times = artificial_spike_times(n_spikes)

    c = FeatureView()
    c.visual.features = features
    c.visual.masks = masks
    c.visual.dimensions = [(0, 0), (1, 0), (2, 0)]
    c.visual.spike_clusters = spike_clusters
    c.visual.spike_times = spike_times
    c.visual.cluster_colors = np.array([_random_color()
                                        for _ in range(n_clusters)])

    show_test(c, 0)


def test_features_empty():
    _test_features(n_spikes=0, n_clusters=0)


def test_features_full():
    _test_features(n_spikes=200, n_clusters=3)
