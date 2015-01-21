# -*- coding: utf-8 -*-

"""Clustering structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import namedtuple, defaultdict, OrderedDict

import numpy as np

from ...ext.six import iterkeys, itervalues
from ._utils import _unique, _spikes_in_clusters
from ._history import History


#------------------------------------------------------------------------------
# Clustering class
#------------------------------------------------------------------------------

class Clustering(object):
    """Object representing a mapping from spike to cluster labels."""

    def __init__(self, spike_clusters):
        self._cluster_counts = None
        self._cluster_labels = None
        # Spike -> cluster mapping.
        spike_clusters = np.asarray(spike_clusters)
        self._spike_clusters = spike_clusters
        # Keep a copy of the original spike clusters assignement.
        self._spike_clusters_base = spike_clusters.copy()
        self._undo_stack = History(base_item=(None, None))
        if spike_clusters is not None:
            self.update()

    def update(self):
        """Update the cluster counts and labels."""
        if self._spike_clusters is not None:
            _cluster_counts = np.bincount(self._spike_clusters)
            # The following is much faster than np.unique().
            self._cluster_labels = np.nonzero(_cluster_counts)[0]
            # Only keep the non-empty clusters.
            self._cluster_counts = _cluster_counts[self._cluster_labels]

    @property
    def spike_clusters(self):
        """Mapping spike to cluster labels."""
        return self._spike_clusters

    @spike_clusters.setter
    def spike_clusters(self, value):
        self.assign(slice(None, None, None), value)

    @property
    def cluster_labels(self):
        """Labels of all clusters, sorted by label."""
        return self._cluster_labels

    @cluster_labels.setter
    def cluster_labels(self, value):
        raise NotImplementedError("Relabeling clusters has not been "
                                  "implemented yet.")

    @property
    def cluster_counts(self):
        """Number of spikes in each cluster."""
        return self._cluster_counts

    def new_cluster_label(self):
        """Return a new cluster label."""
        return np.max(self._cluster_labels) + 1

    @property
    def n_clusters(self):
        """Number of different clusters."""
        return len(self._cluster_labels)

    # Actions
    #--------------------------------------------------------------------------

    def merge(self, cluster_labels, to=None):
        """Merge several clusters to a new cluster.

        Return the modified spikes.

        """
        if to is None:
            to = self.new_cluster_label()
        # Find all spikes in the specified clusters.
        spikes = _spikes_in_clusters(self.spike_clusters, cluster_labels)
        self.assign(spikes, to)
        return spikes

    def assign(self, spike_labels, cluster_labels):
        """Assign clusters to a number of spikes."""
        self.spike_clusters[spike_labels] = cluster_labels
        self._undo_stack.add((spike_labels, cluster_labels))
        self.update()

    def split(self, spike_labels, to=None):
        """Split a number of spikes into a new cluster."""
        if to is None:
            to = self.new_cluster_label()
        self.assign(spike_labels, to)

    def undo(self):
        """Undo the last cluster assignement operation."""
        self._undo_stack.back()
        # Retrieve the initial spike_cluster structure.
        spike_clusters_new = self._spike_clusters_base.copy()
        # Loop over the history (except the last item because we undo).
        for spike_labels, cluster_labels in self._undo_stack:
            # We update the spike clusters accordingly.
            if spike_labels is not None:
                spike_clusters_new[spike_labels] = cluster_labels
        # Finally, we update the spike clusters.
        # WARNING: we do not call self.assign because we don't want to update
        # the undo stack with this action.
        self._spike_clusters = spike_clusters_new
        self.update()

    def redo(self):
        """Redo the last cluster assignement operation."""
        # Go forward in the stack, and retrieve the new assignement.
        item = self._undo_stack.forward()
        if item is None:
            # No redo has been performed: abort.
            return
        spike_labels, cluster_labels = item
        assert spike_labels is not None
        # We apply the new assignement.
        # WARNING: we do not call self.assign because we don't want to update
        # the undo stack with this action.
        self._spike_clusters[spike_labels] = cluster_labels
        self.update()
