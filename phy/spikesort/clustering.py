# -*- coding: utf-8 -*-

"""Clustering structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six


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
        if spike_clusters is not None:
            self.update()
        # Clustering history.
        self.clear_history()

    def update(self):
        """Update the cluster counts and labels."""
        if self._spike_clusters is not None:
            _cluster_counts = np.bincount(self._spike_clusters)
            self._cluster_labels = np.nonzero(_cluster_counts)[0]
            # Only keep the non-empty clusters.
            self._cluster_counts = _cluster_counts[self._cluster_labels]

    @property
    def spike_clusters(self):
        """Mapping spike to cluster labels."""
        return self._spike_clusters

    @spike_clusters.setter
    def spike_clusters(self, value):
        self._spike_clusters = value
        self.update()

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

    def merge(self, cluster_labels, to=None):
        """Merge several clusters to a new cluster."""
        raise NotImplementedError("Merging has not been implemented yet.")

    def split(self, spike_labels, to=None):
        """Split a number of spikes into a new cluster."""
        raise NotImplementedError("Splitting has not been implemented yet.")

    #--------------------------------------------------------------------------
    # Clustering history

    def clear_history(self):
        """Clear the history and save the current clustering."""
        self._history_start = self._spike_clusters.copy()
        self._history = []
        self._history_index = 0  # index of the next history item

    def _add_history_item(self, spikes_changed, cluster):
        """Add a (spikes_changed, cluster) tuple in the clustering history."""
        assert 0 <= self._history_index <= len(self._history)
        self._history = self._history[:self._history_index]
        self._history.append((spikes_changed, cluster))

    def _apply_history(self, until=None, start_at=0):
        """Apply all history items until a given point in the history."""
        if until is None:
            until = self._history_index
        elif until == 0:
            return
        if start_at > until:
            return
        # Check arguments.
        assert until >= 0
        assert start_at >= 0
        assert start_at <= until
        # Start from the first clustering.
        _spike_clusters = self._history_start.copy()
        # Apply all changes successively.
        for i in range(start_at, until):
            _spikes_changed, cluster = self._history[i]
            _spike_clusters[_spikes_changed] = cluster
        # Return the updated clustering.
        return _spike_clusters

    def undo(self):
        """Undo the last clustering action."""
        if self._history_index <= 0:
            return False
        self._history_index -= 1
        # Apply all clustering changes until the penultimate one.
        self.spike_clusters = self._apply_history(self._history_index)
        return True

    def redo(self):
        """Redo the last clustering action."""
        if self._history_index >= len(self._history):
            return False
        self._history_index += 1
        # Apply the latest change.
        self.spike_clusters = self._apply_history(self._history_index,
                                                  self._history_index - 1)
        return True
