# -*- coding: utf-8 -*-

"""Clustering utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _unique(x):
    """Faster version of np.unique().

    This version is restricted to 1D arrays of non-negative integers.

    It is only faster if len(x) >> len(unique(x)).

    """
    if len(x) == 0:
        return np.array([], dtype=np.int)
    return np.nonzero(np.bincount(x))[0]


def _spikes_in_clusters(spike_clusters, clusters):
    """Return the ids of all spikes belonging to the specified clusters."""
    if len(spike_clusters) == 0 or len(clusters) == 0:
        return np.array([], dtype=np.int)
    return np.nonzero(np.in1d(spike_clusters, clusters))[0]


def _spikes_per_cluster(spike_ids, spike_clusters):
    """Return a dictionary {cluster: list_of_spikes}."""
    rel_spikes = np.argsort(spike_clusters)
    abs_spikes = spike_ids[rel_spikes]
    spike_clusters = spike_clusters[rel_spikes]

    diff = np.empty_like(spike_clusters)
    diff[0] = 1
    diff[1:] = np.diff(spike_clusters)

    idx = np.nonzero(diff > 0)[0]
    clusters = spike_clusters[idx]

    spikes_in_clusters = {clusters[i]: np.sort(abs_spikes[idx[i]:idx[i+1]])
                          for i in range(len(clusters) - 1)}
    spikes_in_clusters[clusters[-1]] = np.sort(abs_spikes[idx[-1]:])

    return spikes_in_clusters


def _flatten_spikes_per_cluster(spikes_per_cluster):
    """Convert a dictionary {cluster: list_of_spikes} to a
    spike_clusters array."""
    clusters = sorted(spikes_per_cluster)
    clusters_arr = np.concatenate([(cluster *
                                   np.ones(len(spikes_per_cluster[cluster])))
                                   for cluster in clusters]).astype(np.int64)
    spikes_arr = np.concatenate([spikes_per_cluster[cluster]
                                 for cluster in clusters])
    spike_clusters = np.vstack((spikes_arr, clusters_arr))
    ind = np.argsort(spike_clusters[0, :])
    return spike_clusters[1, ind]
