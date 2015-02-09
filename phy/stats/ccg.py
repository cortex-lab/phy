# -*- coding: utf-8 -*-

"""Cross-correlograms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six
from ..utils.array import _index_of, _unique, _as_array


#------------------------------------------------------------------------------
# Cross-correlograms
#------------------------------------------------------------------------------

def _increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    arr = _as_array(arr)
    indices = _as_array(indices)
    bbins = np.bincount(indices)
    arr[:len(bbins)] += bbins
    return arr


def _diff_shifted(arr, steps=1):
    arr = _as_array(arr)
    return arr[steps:] - arr[:len(arr)-steps]


def _create_correlograms_array(n_clusters, winsize_bins):
    return np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1),
                    dtype=np.int32)


def correlograms(spike_times, spike_clusters,
                 binsize=None, winsize_bins=None):
    """Compute all pairwise cross-correlograms among the clusters appearing
    in 'spike_clusters'.

    Parameters
    ----------

    spike_times : array-like
        Spike times in samples (integers).
    spike_clusters : array-like
        Spike-cluster mapping.
    binsize : int
        Number of time samples in one bin.
    winsize_bins : int
        Number of bins in the window.

    Returns
    -------

    correlograms : array
        A (n_clusters, n_clusters, winsize_samples) array with all pairwise
        CCGs.

    Notes
    -----

    If winsize_samples is the (odd) number of time samples in the window
    then:

        winsize_bins = 2 * ((winsize_samples // 2) // binsize) + 1
        assert winsize_bins % 2 == 1

    For performance reasons, it is recommended to compute the CCGs on a subset
    with only a few thousands or tens of thousands of spikes.

    """

    spike_clusters = _as_array(spike_clusters)
    spike_times = _as_array(spike_times)

    assert spike_times.ndim == 1
    assert spike_times.shape == spike_clusters.shape

    assert winsize_bins % 2 == 1

    clusters = _unique(spike_clusters)
    n_clusters = len(clusters)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _index_of(spike_clusters, clusters)

    # Shift between the two copies of the spike trains.
    shift = 1

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_times, dtype=np.bool)

    correlograms = _create_correlograms_array(n_clusters, winsize_bins)

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _diff_shifted(spike_times, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = spike_diff // binsize

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b > (winsize_bins//2)] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        d = spike_diff_b[m]

        # # Update the masks given the clusters to update.
        # m0 = np.in1d(spike_clusters[:-shift], clusters)
        # m = m & m0
        # d = spike_diff_b[m]
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike clusters.
        indices = np.ravel_multi_index((spike_clusters_i[:-shift][m],
                                        spike_clusters_i[shift:][m], d),
                                       correlograms.shape)

        # Increment the matching spikes in the correlograms array.
        _increment(correlograms.ravel(), indices)

        shift += 1

    return correlograms
