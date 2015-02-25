# -*- coding: utf-8 -*-

"""Clustering structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import namedtuple, defaultdict, OrderedDict
from copy import deepcopy

import numpy as np

from ...ext.six import iterkeys, itervalues, iteritems
from ...utils.array import _as_array
from ._utils import _unique, _spikes_in_clusters
from ._update_info import UpdateInfo
from ._history import History


#------------------------------------------------------------------------------
# Clustering class
#------------------------------------------------------------------------------

def _empty_cluster_counts():
    return defaultdict(lambda: 0)


def _non_empty(cluster_counts):
    clusters = sorted(iterkeys(cluster_counts))
    for cluster in clusters:
        if cluster_counts[cluster] == 0:
            del cluster_counts[cluster]
    return cluster_counts


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
    spike_clusters = _as_array(spike_clusters)
    # Reinitializes the counter.
    _cluster_counts = _empty_cluster_counts()
    # Count the number of spikes in each cluster.
    cluster_counts = np.bincount(spike_clusters)
    # The following is much faster than np.unique().
    clusters_ids = np.nonzero(cluster_counts)[0]
    # Update the counter.
    for cluster in clusters_ids:
        _cluster_counts[cluster] = cluster_counts[cluster]
    return _cluster_counts


class Clustering(object):
    """Object representing a mapping from spike to cluster ids."""

    def __init__(self, spike_clusters):
        self._undo_stack = History(base_item=(None, None))
        # Spike -> cluster mapping.
        self._spike_clusters = _as_array(spike_clusters)
        # Update the cluster counts.
        self.update_cluster_counts()
        # Keep a copy of the original spike clusters assignement.
        self._spike_clusters_base = self._spike_clusters.copy()

    def update_cluster_counts(self):
        """Update the cluster counts and ids."""
        self._cluster_counts = _count_clusters(self._spike_clusters)

    @property
    def spike_clusters(self):
        """Mapping spike to cluster ids."""
        return self._spike_clusters

    @property
    def cluster_ids(self):
        """Labels of all non-empty clusters, sorted by id."""
        return [cluster
                for cluster in sorted(_non_empty(self._cluster_counts))]

    @property
    def cluster_counts(self):
        """Number of spikes in each cluster."""
        return self._cluster_counts

    def new_cluster_id(self):
        """Return a new cluster id."""
        return np.max(self.cluster_ids) + 1

    @property
    def n_clusters(self):
        """Number of different clusters."""
        return len(self.cluster_ids)

    def spikes_in_clusters(self, clusters):
        """Return the spikes belonging to a set of clusters."""
        return _spikes_in_clusters(self.spike_clusters, clusters)

    # Actions
    #--------------------------------------------------------------------------

    def merge(self, cluster_ids, to=None):
        """Merge several clusters to a new cluster."""

        if to is None:
            # Find the new cluster number.
            to = self.new_cluster_id()
        if to < self.new_cluster_id():
            raise ValueError("The new cluster numbers should be higher than "
                             "{0}.".format(self.new_cluster_id()))

        # Find all spikes in the specified clusters.
        spike_ids = _spikes_in_clusters(self.spike_clusters, cluster_ids)

        # Create the UpdateInfo instance here, it's faster.
        update_info = UpdateInfo(description='merge',
                                 clusters=sorted(cluster_ids),
                                 spikes=spike_ids,
                                 added=[to],
                                 deleted=sorted(cluster_ids))

        # And update the cluster counts directly.
        n_spikes = len(spike_ids)
        for cluster in cluster_ids:
            if cluster in self._cluster_counts:
                del self._cluster_counts[cluster]
        self._cluster_counts[to] = n_spikes

        # NOTE: we could have called self.assign() here, but we don't.
        # We circumvent self.assign() for performance reasons.
        # assign() is a relatively costly operation, whereas merging is a much
        # cheaper operation.

        # Assign the clusters.
        self.spike_clusters[spike_ids] = to

        # Add to stack.
        self._undo_stack.add((spike_ids, [to]))

        return update_info

    def assign(self, spike_ids, cluster_ids,
               _add_to_stack=True):
        """Assign clusters to a number of spikes."""
        # Ensure 'cluster_ids' is an array-like.
        if not hasattr(cluster_ids, '__len__'):
            cluster_ids = [cluster_ids]
        # Check the sizes of spike_ids and cluster_ids.
        if (isinstance(spike_ids, (np.ndarray, list)) and
           len(cluster_ids) > 1):
            assert len(spike_ids) == len(cluster_ids)
        self.spike_clusters[spike_ids] = cluster_ids
        # If the UpdateInfo is passed, it means the _cluster_counts structure
        # has already been updated. Otherwise, we need to update it here.
        counts_before = self._cluster_counts
        self.update_cluster_counts()
        counts_after = self._cluster_counts
        update_info = _diff_counts(counts_before, counts_after)
        update_info.description = 'assign'
        update_info.spikes = spike_ids
        if _add_to_stack:
            self._undo_stack.add((spike_ids, cluster_ids))
        assert update_info is not None
        return update_info

    def split(self, spike_ids):
        """Split a number of spikes into a new cluster."""
        # self.assign() accepts relative numbers as second argument.
        return self.assign(spike_ids, self.new_cluster_id())

    def undo(self):
        """Undo the last cluster assignement operation."""
        self._undo_stack.back()
        # Retrieve the initial spike_cluster structure.
        spike_clusters_new = self._spike_clusters_base.copy()
        # Loop over the history (except the last item because we undo).
        for spike_ids, cluster_ids in self._undo_stack:
            # We update the spike clusters accordingly.
            if spike_ids is not None:
                spike_clusters_new[spike_ids] = cluster_ids
        # Finally, we update all spike clusters.
        # WARNING: do not add an item in the stack.
        return self.assign(slice(None, None, None), spike_clusters_new,
                           _add_to_stack=False)

    def redo(self):
        """Redo the last cluster assignement operation."""
        # Go forward in the stack, and retrieve the new assignement.
        item = self._undo_stack.forward()
        if item is None:
            # No redo has been performed: abort.
            return
        spike_ids, cluster_ids = item
        assert spike_ids is not None
        # We apply the new assignement.
        # WARNING: do not add an item in the stack.
        return self.assign(spike_ids, cluster_ids,
                           _add_to_stack=False)
