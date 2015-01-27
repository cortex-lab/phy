# -*- coding: utf-8 -*-

"""Cross-correlograms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.lib.stride_tricks import as_strided

from ..ext import six


#------------------------------------------------------------------------------
# Cross-correlograms
#------------------------------------------------------------------------------

"""

Variables
---------

winsize_samples : int
    Number of time samples in the window. Needs to be an odd number.

binsize : int
    Number of time samples in one bin.

winsize_bins : int
    Number of bins in the window.

    winsize_bins = 2 * ((winsize_samples // 2) // binsize) + 1
    assert winsize_bins % 2 == 1

"""


def _increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    arr = np.asarray(arr)
    indices = np.asarray(indices)
    bbins = np.bincount(indices)
    arr[:len(bbins)] += bbins
    return arr


def _diff_shifted(arr, steps=1):
    arr = np.asarray(arr)
    return arr[steps:] - arr[:len(arr)-steps]


def _create_correlograms_array(n_clusters, winsize_bins):
    return np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1),
                    dtype=np.int32)


class Correlograms(object):
    def __init__(self, spike_times, binsize=None, winsize_bins=None):
        """We compute a few data structures at initialization time, as
        these can be reused for the computation of the CCGs with different
        clusterings."""
        self.spike_times = spike_times
        if binsize is not None and winsize_bins is not None:
            self.initialize(binsize, winsize_bins)

    def initialize(self, binsize, winsize_bins):
        """Need to be called whenever binsize of winsize_bins changes."""
        self.binsize = binsize
        self.winsize_bins = winsize_bins

        # Internal cache that can be reused during manual clustering.
        self._cache = []

        shift = 1  # Shift between the two copies of the spike trains.

        # At a given shift, the mask precises which spikes have matching spikes
        # within the correlogram time window.
        mask = np.ones_like(self.spike_times, dtype=np.bool)

        # The loop continues as long as there is at least one spike with
        # a matching spike.
        while mask[:-shift].any():
            # Number of time samples between spike i and spike i+shift.
            spikediff = _diff_shifted(self.spike_times, shift)

            # Binarize the delays between spike i and spike i+shift.
            spikediff_b = spikediff // binsize

            # Spikes with no matching spikes are masked.
            mask[:-shift][spikediff_b > (self.winsize_bins//2)] = False

            # Cache the masked spike delays.
            m = mask[:-shift].copy()
            d = spikediff_b[m]

            self._cache.append((shift, m, d, spikediff_b))

            shift += 1

    def compute(self, spike_clusters, clusters):
        """Compute pairwise CCGs.

        The clusters which CCGs need to be recomputed can be precised.

        Assume `initialize()` has been called if binsize, or winsize_bins
        have changed.

        Arguments
        ---------

        spike_clusters : array-like
            Spike-cluster mapping.

        clusters : array-like
            List of clusters to compute the CCGs from.

        """

        assert clusters is not None
        clusters = np.asarray(clusters)
        n_clusters = len(clusters)
        assert n_clusters > 0

        correlograms = _create_correlograms_array(n_clusters,
                                                  self.winsize_bins)

        # Loop over all shifts.
        for shift, m, d, spikediff_b in self._cache:

            # Update the masks given the clusters to update.
            m0 = np.in1d(spike_clusters[:-shift], clusters)
            m = m & m0
            d = spikediff_b[m]

            # Find the indices in the raveled correlograms array that need
            # to be incremented, taking into account the spike clusters.
            indices = np.ravel_multi_index((spike_clusters[:-shift][m],
                                            spike_clusters[shift:][m], d),
                                           correlograms.shape)

            # Increment the matching spikes in the correlograms array.
            _increment(correlograms.ravel(), indices)

        return correlograms

    def merged(self, clusters, to):
        """Efficiently update the CCGs after a merge."""
        raise NotImplementedError()
