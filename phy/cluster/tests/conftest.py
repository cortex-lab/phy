# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from pytest import fixture

from phylib.io.array import get_closest_clusters
import phy.gui.qt

# Reduce the debouncer delay for tests.
phy.gui.qt.Debouncer.delay = 1


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def cluster_ids():
    return [0, 1, 2, 10, 11, 20, 30]
    #       i, g, N,  i,  g,  N, N


@fixture
def spike_clusters(cluster_ids):
    return np.repeat(cluster_ids, 2 + np.arange(len(cluster_ids)))


@fixture
def cluster_groups():
    return {0: 'noise', 1: 'good', 10: 'mua', 11: 'good'}


@fixture
def cluster_labels():
    return {'test_label': {10: 123, 0: 456}, 'group': {}}


@fixture
def cluster_metrics():
    n_spikes = {0: 2, 1: 3, 2: 4, 10: 5, 11: 6, 20: 7, 30: 8}
    return {'n_spikes': lambda cl: n_spikes.get(cl, None)}


@fixture
def similarity(cluster_ids):
    sim = lambda c, d: (c * 1.01 + d)

    def similarity(selected):
        if not selected:
            return []
        out = get_closest_clusters(selected[0], cluster_ids, sim)
        out = [(c, s) for c, s in out if c not in selected]
        return out
    return similarity
