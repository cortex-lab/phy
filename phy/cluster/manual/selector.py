# -*- coding: utf-8 -*-

"""Selector structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...ext import six
from ._utils import _unique, _spikes_in_clusters
from ...utils.logging import debug, info, warn


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
        # Update the selected spikes accordingly.
        self.selected_spikes = self._subset()
        if self._n_spikes_max is not None:
            assert len(self._selected_spikes) <= self._n_spikes_max

    def _subset(self, spikes=None, n_spikes_max=None):
        """Prune the current selection to get at most n_spikes_max spikes."""
        if n_spikes_max is None:
            n_spikes_max = self._n_spikes_max
        if spikes is None:
            spikes = self._selected_spikes
        # Nothing to do if the selection already satisfies n_spikes_max.
        if n_spikes_max is None or len(spikes) <= n_spikes_max:
            return spikes
        # Fill 50% regularly sampled spikes for the selection.
        step = int(np.clip(2. / n_spikes_max * len(spikes),
                           1, len(spikes)))
        my_spikes = spikes[::step]
        assert len(my_spikes) <= len(spikes)
        assert len(my_spikes) <= n_spikes_max
        # Number of remaining spikes to find in the selection.
        n_start = (n_spikes_max - len(my_spikes)) // 2
        n_end = n_spikes_max - len(my_spikes) - n_start
        assert (n_start >= 0) & (n_end >= 0)
        # The other 50% come from the start and end of the selection.
        my_spikes = np.r_[spikes[:n_start],
                          my_spikes,
                          spikes[-n_end:]]
        my_spikes = _unique(my_spikes)
        assert len(my_spikes) <= n_spikes_max
        return my_spikes

    @property
    def selected_spikes(self):
        """Labels of the selected spikes."""
        return self._selected_spikes

    @selected_spikes.setter
    def selected_spikes(self, value):
        """Explicitely select a number of spikes."""
        value = np.asarray(value)
        # Make sure there are less spikes than n_spikes_max.
        self._selected_spikes = self._subset(value)

    @property
    def selected_clusters(self):
        """Clusters containing at least one selected spike."""
        return _unique(self._spike_clusters[self._selected_spikes])

    @selected_clusters.setter
    def selected_clusters(self, value):
        """Select spikes belonging to a number of clusters."""
        # TODO: smarter subselection: select n_spikes_max/n_clusters spikes
        # per cluster, so that the number of spikes per cluster is independent
        # from the sizes of the clusters.
        value = np.asarray(value)
        # All spikes from the selected clusters.
        spikes = _spikes_in_clusters(self._spike_clusters, value)
        # Make sure there are less spikes than n_spikes_max.
        self.selected_spikes = self._subset(spikes)
