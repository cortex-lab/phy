# -*- coding: utf-8 -*-

"""Clustering structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six


#------------------------------------------------------------------------------
# History class
#------------------------------------------------------------------------------

class History(object):
    """Implement a history of actions with an undo stack."""
    def __init__(self):
        self.clear()

    def clear(self):
        """Clear the history."""
        self._history = []
        self._index = 0  # index of the next history item

    @property
    def current_item(self):
        """Return the current element."""
        if self._history and self._index >= 1:
            self._check_index()
            return self._history[self._index - 1]

    @property
    def current_position(self):
        """Current position in the history."""
        if self._index >= 1:
            return self._index - 1

    def _check_index(self):
        """Check that the index is without the bounds of _history."""
        assert 0 <= self._index <= len(self._history)

    def add(self, item):
        """Add an item in the history."""
        self._check_index()
        # Possibly truncate the history up to the current point.
        self._history = self._history[:self._index]
        # Append the item
        self._history.append(item)
        # Increment the index.
        self._index += 1
        self._check_index()
        # Check that the current element is what was provided to the function.
        assert id(self.current_item) == id(item)

    def iter(self, until=None, start_at=0):
        """Iterate through successive history items."""
        if until is None:
            until = self._index
        elif until == 0:
            return
        if start_at >= until:
            return
        # Check arguments.
        assert until >= 0
        assert start_at >= 0
        assert start_at < until
        for i in range(start_at, until):
            yield self._history[i]

    def __iter__(self):
        return self.iter()

    def __len__(self):
        return len(self._history)

    def back(self):
        """Go back in history if possible."""
        if self._index <= 0:
            return False
        self._index -= 1
        self._check_index()
        return True

    def forward(self):
        """Go forward in history if possible."""
        if self._index >= len(self._history):
            return False
        self._index += 1
        self._check_index()
        return True


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
