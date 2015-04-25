# -*- coding: utf-8 -*-

"""Test CCG plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.logging import set_level
from ..traces import TraceView
from ...utils._color import _random_color
from ...io.mock.artificial import (artificial_traces,
                                   artificial_masks,
                                   artificial_spike_clusters,
                                   )
from ...utils.testing import show_test


#------------------------------------------------------------------------------
# Tests VisPy
#------------------------------------------------------------------------------

def _test_traces(n_samples=None):
    n_channels = 20
    n_spikes = 10
    n_clusters = 3

    traces = artificial_traces(n_samples, n_channels)
    masks = artificial_masks(n_spikes, n_channels)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    c = TraceView()
    c.visual.traces = traces
    c.visual.channel_colors = np.array([_random_color()
                                        for _ in range(n_channels)])
    c.visual.spike_ids = np.arange(n_spikes)
    c.visual.spike_clusters = spike_clusters
    c.visual.cluster_colors = np.array([_random_color()
                                        for _ in range(n_clusters)])
    c.visual.masks = masks

    show_test(c, 0)


def test_traces_empty():
    _test_traces(n_samples=0)


def test_traces_full():
    _test_traces(n_samples=1000)
