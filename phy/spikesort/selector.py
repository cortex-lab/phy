# -*- coding: utf-8 -*-

"""Selector structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six
from ..utils.logging import debug, info, warn


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _unique(x):
    """Faster version of np.unique().

    This version is restricted to 1D arrays of non-negative integers.

    It is only faster if len(x) >> len(unique(x)).

    """
    return np.nonzero(np.bincount(x))[0]


def _spikes_in_clusters(spike_clusters, clusters):
    """Return the labels of all spikes belonging to the specified clusters."""
    return np.nonzero(np.in1d(spike_clusters, clusters))[0]


#------------------------------------------------------------------------------
# Selector class
#------------------------------------------------------------------------------

class Selector(object):
    """Object representing a selection of spikes or clusters."""
    def __init__(self, spike_clusters, n_spikes_max=None):
        self._spike_clusters = spike_clusters
        self._n_spikes_max = n_spikes_max
        self._selected_spikes = np.array([], dtype=np.int64)

    @property
    def n_spikes_max(self):
        """Maximum number of spikes allowed in the selection."""
        return self._n_spikes_max

    @n_spikes_max.setter
    def n_spikes_max(self, value):
        self._n_spikes_max = value

    @property
    def selected_spikes(self):
        """Labels of the selected spikes."""
        return self._selected_spikes

    @selected_spikes.setter
    def selected_spikes(self, value):
        """Explicitely select a number of spikes."""
        value = np.asarray(value)
        size = len(value)
        size_max = self._n_spikes_max
        if self._n_spikes_max is not None and size > size_max:
            debug("{0:d} spikes were selected whereas ".format(size) +
                  "no more than {0:d} are allowed; ".format(size_max) +
                  "keeping only the first {0:d} now.".format(size_max))
            self._selected_spikes = value[:size_max]
        else:
            self._selected_spikes = value

    @property
    def selected_clusters(self):
        """Clusters containing at least one selected spike."""
        return _unique(self._spike_clusters[self._selected_spikes])

    @selected_clusters.setter
    def selected_clusters(self, value):
        """Select spikes belonging to a number of clusters."""
        value = np.asarray(value)
        all_spikes = _spikes_in_clusters(self._spike_clusters, value)
        # Select all spikes from the selected clusters.
        if ((self._n_spikes_max is None) or
           (len(all_spikes) <= self._n_spikes_max)):
            self.selected_spikes = all_spikes
        else:
            # Select a carefully chosen subset of spikes from the selected
            # clusters.
            # Fill 50% regularly sampled spikes for the selection.
            n_max = self._n_spikes_max
            step = int(np.clip(2. / n_max * len(all_spikes),
                               1, len(all_spikes)))
            my_spikes = all_spikes[::step]
            n_rest = n_max - len(my_spikes)
            # The other 50% come from the start and end of the selection.
            my_spikes = np.r_[all_spikes[:n_rest // 2],
                              my_spikes,
                              all_spikes[-(n_max - nrest // 2):]]
            assert len(my_spikes) == n_max
            self.selected_spikes = np.sort(my_spikes)
