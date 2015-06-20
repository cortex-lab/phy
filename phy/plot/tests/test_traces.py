# -*- coding: utf-8 -*-

"""Test CCG plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

import numpy as np

from ..traces import TraceView, plot_traces
from ...utils._color import _random_color
from ...io.mock import (artificial_traces,
                        artificial_masks,
                        artificial_spike_clusters,
                        )
from ...utils.testing import show_test


# Skip these tests in "make test-quick".
pytestmark = mark.long()


#------------------------------------------------------------------------------
# Tests VisPy
#------------------------------------------------------------------------------

def _test_traces(n_samples=None):
    n_channels = 20
    n_spikes = 50
    n_clusters = 3

    traces = artificial_traces(n_samples, n_channels)
    masks = artificial_masks(n_spikes, n_channels)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_samples = np.linspace(50, n_samples - 50, n_spikes).astype(np.uint64)

    c = TraceView(keys='interactive')
    c.visual.traces = traces
    c.visual.n_samples_per_spike = 20
    c.visual.spike_samples = spike_samples
    c.visual.spike_clusters = spike_clusters
    c.visual.cluster_colors = np.array([_random_color()
                                        for _ in range(n_clusters)])
    c.visual.masks = masks
    c.visual.sample_rate = 20000.
    c.visual.offset = 0

    show_test(c)


def test_traces_empty():
    _test_traces(n_samples=0)


def test_traces_full():
    _test_traces(n_samples=2000)


def test_plot_traces():
    n_samples = 10000
    n_channels = 20
    n_spikes = 50
    n_clusters = 3

    traces = artificial_traces(n_samples, n_channels)
    masks = artificial_masks(n_spikes, n_channels)
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    spike_samples = np.linspace(50, n_samples - 50, n_spikes).astype(np.uint64)

    c = plot_traces(traces, show=False)
    show_test(c)

    c = plot_traces(traces,
                    spike_samples=spike_samples,
                    masks=masks, show=False)
    show_test(c)

    c = plot_traces(traces,
                    spike_samples=spike_samples,
                    spike_clusters=spike_clusters, show=False)
    show_test(c)

    c = plot_traces(traces,
                    spike_samples=spike_samples,
                    spike_clusters=spike_clusters,
                    show=False)
    show_test(c)
