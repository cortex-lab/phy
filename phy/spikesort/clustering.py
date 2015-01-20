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

    def __init__(self, spike_clusters=None):
        self._cluster_counts = None
        self._cluster_labels = None
        self._spike_clusters = spike_clusters
        if spike_clusters is not None:
            self.update()

    def update(self):
        """Update the cluster counts and labels."""
        if self._spike_clusters is not None:
            self._cluster_counts = np.bincount(self._spike_clusters)
            self._cluster_labels = np.nonzero(self._cluster_counts)[0]

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

    def new_cluster_label(self):
        """Return a new cluster label."""
        return np.max(self._cluster_labels) + 1

    @property
    def n_clusters(self):
        """Number of different clusters."""
        return len(self._cluster_labels)

    def merge(cluster_labels, to=None):
        """Merge several clusters to a new cluster."""
        raise NotImplementedError("Merging has not been implemented yet.")

    def split(spike_labels, to=None):
        """Split a number of spikes into a new cluster."""
        raise NotImplementedError("Splitting has not been implemented yet.")
