# -*- coding: utf-8 -*-

"""Clustering structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import namedtuple, defaultdict, OrderedDict
from copy import deepcopy

import numpy as np

from ...ext.six import iterkeys, itervalues, iteritems
from ._utils import _unique, _spikes_in_clusters
from ._update_info import UpdateInfo
from ._history import History


#------------------------------------------------------------------------------
# Clustering class
#------------------------------------------------------------------------------

def _empty_cluster_counts():
    return defaultdict(lambda: 0)


def _diff_counts(count_1, count_2):
    # List of all non-empty clusters before and after.
    clusters_before = set(_non_empty(count_1))
    clusters_after = set(_non_empty(count_2))
    # Added and deleted clusters.
    added = clusters_after - clusters_before
    deleted = clusters_before - clusters_after
    # Find the clusters that have their counts changed.
    intersection = clusters_before.intersection(clusters_after)
    count_changed = [cluster for cluster in sorted(intersection)
                     if count_1[cluster] != count_2[cluster]]
    # Return the UpdateInfo object.
    update_info = UpdateInfo(added=sorted(added),
                             deleted=sorted(deleted),
                             count_changed=sorted(count_changed))
    return update_info


def _count_clusters(spike_clusters):
    """Compute cluster counts."""
    # Reinitializes the counter.
    _cluster_counts = _empty_cluster_counts()
    # Count the number of spikes in each cluster.
    cluster_counts = np.bincount(spike_clusters)
    # The following is much faster than np.unique().
    clusters_labels = np.nonzero(cluster_counts)[0]
    # Update the counter.
    for cluster in clusters_labels:
        _cluster_counts[cluster] = cluster_counts[cluster]
    return _cluster_counts


def _non_empty(cluster_counts):
    clusters = sorted(iterkeys(cluster_counts))
    for cluster in clusters:
        if cluster_counts[cluster] == 0:
            del cluster_counts[cluster]
    return cluster_counts


def _get_update_info(spike_labels,
                     cluster_counts_before, cluster_counts_after):
    """Return an UpdateInfo instance as a function of new spike->cluster
    assignements."""
    update_info = _diff_counts(cluster_counts_before, cluster_counts_after)
    update_info.spikes = spike_labels
    return update_info


class Clustering(object):
    """Object representing a mapping from spike to cluster labels."""

    def __init__(self, spike_clusters):
        self._undo_stack = History(base_item=(None, None))
        # Spike -> cluster mapping.
        self._spike_clusters = np.asarray(spike_clusters)
        # Update the cluster counts.
        self.update_cluster_counts()
        # Keep a copy of the original spike clusters assignement.
        self._spike_clusters_base = self._spike_clusters.copy()

    def update_cluster_counts(self):
        """Update the cluster counts and labels."""
        self._cluster_counts = _count_clusters(self._spike_clusters)

    @property
    def spike_clusters(self):
        """Mapping spike to cluster labels."""
        return self._spike_clusters

    @property
    def cluster_labels(self):
        """Labels of all non-empty clusters, sorted by label."""
        return [cluster
                for cluster in sorted(_non_empty(self._cluster_counts))]

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
        return np.max(self.cluster_labels) + 1

    @property
    def n_clusters(self):
        """Number of different clusters."""
        return len(self.cluster_labels)

    # Actions
    #--------------------------------------------------------------------------

    def merge(self, cluster_labels, to=None):
        """Merge several clusters to a new cluster."""
        if to is None:
            to = self.new_cluster_label()
        # Find all spikes in the specified clusters.
        spikes = _spikes_in_clusters(self.spike_clusters, cluster_labels)
        # Create the UpdateInfo instance here, it's faster.
        _update_info = UpdateInfo(description='merge',
                                  clusters=cluster_labels,
                                  spikes=spikes,
                                  added=[to])
        # And update the cluster counts directly.
        n_spikes = len(spikes)
        # This is just for debugging.
        # n_spikes_bis = sum([self._cluster_counts[cluster]
        #                     for cluster in cluster_labels])
        # assert n_spikes_bis == n_spikes
        for cluster in cluster_labels:
            del self._cluster_counts[cluster]
        self._cluster_counts[to] = n_spikes
        # Finally, assign the spike clusters and return directly the
        # UpdateInfo instance.
        return self.assign(spikes, to, _update_info=_update_info)

    def _assign(self, spike_labels, cluster_labels, _update_info=None):
        """Assign clusters to a number of spikes, but do not add
        the change to the undo stack."""
        # Ensure 'cluster_labels' is an array-like.
        if not hasattr(cluster_labels, '__len__'):
            cluster_labels = [cluster_labels]
        self.spike_clusters[spike_labels] = cluster_labels
        # If the UpdateInfo is passed, it means the _cluster_counts structure
        # has already been updated. Otherwise, we need to update it here.
        if _update_info is None:
            counts_before = self._cluster_counts
            self.update_cluster_counts()
            counts_after = self._cluster_counts
            _update_info = _get_update_info(spike_labels,
                                            counts_before, counts_after)
        return _update_info

    def assign(self, spike_labels, cluster_labels, _update_info=None):
        """Assign clusters to a number of spikes."""
        up = self._assign(spike_labels, cluster_labels, _update_info)
        self._undo_stack.add((spike_labels, cluster_labels))
        return up

    def split(self, spike_labels, to=None):
        """Split a number of spikes into a new cluster."""
        if to is None:
            to = self.new_cluster_label()
        return self.assign(spike_labels, to)

    def undo(self):
        """Undo the last cluster assignement operation."""
        self._undo_stack.back()
        # Retrieve the initial spike_cluster structure.
        spike_clusters_new = self._spike_clusters_base.copy()
        # This structure contains True when the spike has been updated.
        # spike_changes = np.zeros_like(spike_clusters_new, dtype=np.bool)
        # Loop over the history (except the last item because we undo).
        for spike_labels, cluster_labels in self._undo_stack:
            # We update the spike clusters accordingly.
            if spike_labels is not None:
                spike_clusters_new[spike_labels] = cluster_labels
                # spike_changes[spike_labels] = True
        # spike_changed = np.nonzero(spike_changes)[0]
        # Finally, we update all spike clusters.
        # WARNING: do not add an item in the stack (_assign and not assign).
        return self._assign(slice(None, None, None), spike_clusters_new)

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
        # WARNING: do not add an item in the stack (_assign and not assign).
        return self._assign(spike_labels, cluster_labels)
