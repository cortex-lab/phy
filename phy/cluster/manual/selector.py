# -*- coding: utf-8 -*-

"""Selector structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.array import _as_array, regular_subset, get_excerpts
from ._utils import _unique, _spikes_in_clusters


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
        self.selected_spikes = self.subset_spikes()
        if self._n_spikes_max is not None:
            assert len(self._selected_spikes) <= self._n_spikes_max

    def subset_spikes(self,
                      spikes=None,
                      n_spikes_max=None,
                      excerpt_size=None,
                      ):
        """Prune the current selection to get at most n_spikes_max spikes.

        If excerpt_size is None, return a regular strided selection.
        Otherwise, return a regular selection of contiguous chunks.

        """
        if spikes is None:
            spikes = self._selected_spikes
        if n_spikes_max is None:
            n_spikes_max = self._n_spikes_max
        if excerpt_size is None:
            return regular_subset(spikes, n_spikes_max)
        else:
            n_excerpts = n_spikes_max // excerpt_size
            return get_excerpts(spikes,
                                n_excerpts=n_excerpts,
                                excerpt_size=excerpt_size,
                                )

    @property
    def selected_spikes(self):
        """Labels of the selected spikes."""
        return self._selected_spikes

    @selected_spikes.setter
    def selected_spikes(self, value):
        """Explicitely select a number of spikes."""
        value = _as_array(value)
        # Make sure there are less spikes than n_spikes_max.
        self._selected_spikes = self.subset_spikes(value)

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
        value = _as_array(value)
        # All spikes from the selected clusters.
        spikes = _spikes_in_clusters(self._spike_clusters, value)
        # Make sure there are less spikes than n_spikes_max.
        self.selected_spikes = self.subset_spikes(spikes)

    def on_cluster(self, up=None):
        """Called when clustering has changed."""
        # TODO
        pass
