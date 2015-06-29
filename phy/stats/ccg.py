# -*- coding: utf-8 -*-

"""Cross-correlograms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..utils._types import _as_array
from ..utils.array import _index_of, _unique


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


def correlograms(spike_samples, spike_clusters,
                 cluster_order=None,
                 binsize=None, winsize_bins=None):
    """Compute all pairwise cross-correlograms among the clusters appearing
    in `spike_clusters`.

    Parameters
    ----------

    spike_samples : array-like
        Spike times in samples (integers).
    spike_clusters : array-like
        Spike-cluster mapping.
    cluster_order : array-like
        The list of unique clusters, in any order. That order will be used
        in the output array.
    binsize : int
        Number of time samples in one bin.
    winsize_bins : int (odd number)
        Number of bins in the window.

    Returns
    -------

    correlograms : array
        A `(n_clusters, n_clusters, winsize_samples)` array with all pairwise
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
    spike_samples = _as_array(spike_samples)
    if spike_samples.dtype in (np.int32, np.int64):
        spike_samples = spike_samples.astype(np.uint64)

    assert spike_samples.dtype == np.uint64

    assert spike_samples.ndim == 1
    assert spike_samples.shape == spike_clusters.shape

    assert winsize_bins % 2 == 1

    # Take the cluster oder into account.
    if cluster_order is None:
        clusters = _unique(spike_clusters)
    else:
        clusters = _as_array(cluster_order)
    n_clusters = len(clusters)

    # Like spike_clusters, but with 0..n_clusters-1 indices.
    spike_clusters_i = _index_of(spike_clusters, clusters)

    # Shift between the two copies of the spike trains.
    shift = 1

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_samples, dtype=np.bool)

    correlograms = _create_correlograms_array(n_clusters, winsize_bins)

    # The loop continues as long as there is at least one spike with
    # a matching spike.
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = _diff_shifted(spike_samples, shift)

        # Binarize the delays between spike i and spike i+shift.
        spike_diff_b = spike_diff // binsize

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False

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
                                        spike_clusters_i[+shift:][m],
                                        d),
                                       correlograms.shape)

        # Increment the matching spikes in the correlograms array.
        _increment(correlograms.ravel(), indices)

        shift += 1

    # Remove ACG peaks.
    correlograms[np.arange(n_clusters),
                 np.arange(n_clusters),
                 0] = 0

    return correlograms


#------------------------------------------------------------------------------
# Helper functions for CCG data structures
#------------------------------------------------------------------------------

def _symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays."""

    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # We symmetrize c[i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    correlograms[..., 0] = np.maximum(correlograms[..., 0],
                                      correlograms[..., 0].T)

    sym = correlograms[..., 1:][..., ::-1]
    sym = np.transpose(sym, (1, 0, 2))

    return np.dstack((sym, correlograms))


def pairwise_correlograms(spike_samples,
                          spike_clusters,
                          binsize=None,
                          winsize_bins=None,
                          ):
    """Compute all pairwise correlograms in a set of neurons.

    TODO: improve interface and documentation.

    """
    ccgs = correlograms(spike_samples,
                        spike_clusters,
                        binsize=binsize,
                        winsize_bins=winsize_bins,
                        )
    ccgs = _symmetrize_correlograms(ccgs)
    return ccgs
