# -*- coding: utf-8 -*-

"""Clustering structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import namedtuple, defaultdict, OrderedDict
from copy import deepcopy

import numpy as np

from ...ext.six import iterkeys, itervalues, iteritems
from ...utils.array import _as_array, _is_array_like
from ._utils import _unique, _spikes_in_clusters, _spikes_per_cluster
from ._update_info import UpdateInfo
from ._history import History


#------------------------------------------------------------------------------
# Clustering class
#------------------------------------------------------------------------------

def _extend_spikes(spike_ids, spike_clusters):
    """Return all spikes belonging to the clusters containing the specified
    spikes."""
    # We find the spikes belonging to modified clusters.
    # What are the old clusters that are modified by the assignement?
    old_spike_clusters = spike_clusters[spike_ids]
    unique_clusters = _unique(old_spike_clusters)
    # Now we take all spikes from these clusters.
    changed_spike_ids = _spikes_in_clusters(spike_clusters, unique_clusters)
    # These are the new spikes that need to be reassigned.
    extended_spike_ids = np.setdiff1d(changed_spike_ids, spike_ids,
                                      assume_unique=True)
    return extended_spike_ids


def _concatenate_spike_clusters(*pairs):
    """Concatenate a list of pairs (spike_ids, spike_clusters)."""
    pairs = [(_as_array(x), _as_array(y)) for (x, y) in pairs]
    concat = np.vstack(np.hstack((x[:, None], y[:, None]))
                       for x, y in pairs)
    reorder = np.argsort(concat[:, 0])
    concat = concat[reorder, :]
    return concat[:, 0].astype(np.int64), concat[:, 1].astype(np.int64)


def _extend_assignement(spike_ids, old_spike_clusters, spike_clusters_rel):
    # 1. Add spikes that belong to modified clusters.
    # 2. Find new cluster ids for all changed clusters.

    old_spike_clusters = _as_array(old_spike_clusters)
    spike_ids = _as_array(spike_ids)

    assert isinstance(spike_clusters_rel, (list, np.ndarray))
    spike_clusters_rel = _as_array(spike_clusters_rel)
    assert spike_clusters_rel.min() >= 0

    # We renumber the new cluster indices.
    new_cluster_id = old_spike_clusters.max() + 1
    new_spike_clusters = (spike_clusters_rel +
                          (new_cluster_id - spike_clusters_rel.min()))

    # We find the spikes belonging to modified clusters.
    extended_spike_ids = _extend_spikes(spike_ids, old_spike_clusters)
    if len(extended_spike_ids) == 0:
        return spike_ids, new_spike_clusters

    # We take their clusters.
    extended_spike_clusters = old_spike_clusters[extended_spike_ids]
    # Generate new cluster numbers.
    k = new_spike_clusters.max() + 1
    extended_spike_clusters += (k - extended_spike_clusters.min())

    # Finally, we concatenate spike_ids and extended_spike_ids.
    return _concatenate_spike_clusters((spike_ids,
                                        new_spike_clusters),
                                       (extended_spike_ids,
                                        extended_spike_clusters))


def _assign_update_info(spike_ids, old_spike_clusters, new_spike_clusters):
    old_clusters = np.unique(old_spike_clusters)
    new_clusters = np.unique(new_spike_clusters)
    descendants = list(set(zip(old_spike_clusters,
                               new_spike_clusters)))
    update_info = UpdateInfo(description='assign',
                             spikes=spike_ids,
                             descendants=descendants,
                             added=list(new_clusters),
                             deleted=list(old_clusters))
    return update_info


class Clustering(object):
    """Object representing a mapping from spike to cluster ids."""

    def __init__(self, spike_clusters):
        self._undo_stack = History(base_item=(None, None))
        # Spike -> cluster mapping.
        self._spike_clusters = _as_array(spike_clusters)
        # Create the spikes per cluster structure.
        self._update_all_spikes_per_cluster()
        # Keep a copy of the original spike clusters assignement.
        self._spike_clusters_base = self._spike_clusters.copy()

    @property
    def spike_clusters(self):
        """Mapping spike to cluster ids."""
        return self._spike_clusters

    @property
    def spikes_per_cluster(self):
        """Dictionary {cluster: array_of_spikes}."""
        return self._spikes_per_cluster

    @property
    def cluster_ids(self):
        """Labels of all non-empty clusters, sorted by id."""
        return sorted(self._spikes_per_cluster)

    @property
    def cluster_counts(self):
        """Number of spikes in each cluster."""
        return {cluster: len(self._spikes_per_cluster[cluster])
                for cluster in self.cluster_ids}

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

        # Find the new cluster number.
        if to is None:
            to = self.new_cluster_id()
        if to < self.new_cluster_id():
            raise ValueError("The new cluster numbers should be higher than "
                             "{0}.".format(self.new_cluster_id()))

        cluster_ids = sorted(cluster_ids)

        # NOTE: we could have called self.assign() here, but we don't.
        # We circumvent self.assign() for performance reasons.
        # assign() is a relatively costly operation, whereas merging is a much
        # cheaper operation.

        # Find all spikes in the specified clusters.
        spike_ids = _spikes_in_clusters(self.spike_clusters, cluster_ids)

        # Create the UpdateInfo instance here.
        descendants = [(cluster, to) for cluster in cluster_ids]
        update_info = UpdateInfo(description='merge',
                                 spikes=spike_ids,
                                 added=[to],
                                 deleted=cluster_ids,
                                 descendants=descendants
                                 )

        # Update the spikes_per_cluster structure directly.
        self._spikes_per_cluster[to] = spike_ids
        for cluster in cluster_ids:
            del self._spikes_per_cluster[cluster]

        # Assign the clusters.
        self.spike_clusters[spike_ids] = to

        # Add to stack.
        self._undo_stack.add((spike_ids, [to]))

        return update_info

    def _update_all_spikes_per_cluster(self):
        self._spikes_per_cluster = _spikes_per_cluster(self._spike_clusters)

    def _do_assign(self, spike_ids, new_spike_clusters):
        """Make spike-cluster assignements after the spike selection has
        been extended to full clusters."""

        # Ensure spike_clusters has the right shape.
        spike_ids = _as_array(spike_ids)
        if len(new_spike_clusters) == 1 and len(spike_ids) > 1:
            new_spike_clusters = (np.ones(len(spike_ids)) *
                                  new_spike_clusters[0])
        old_spike_clusters = self._spike_clusters[spike_ids]

        assert len(spike_ids) == len(old_spike_clusters)
        assert len(new_spike_clusters) == len(spike_ids)

        # Update the spikes per cluster structure.
        new_spikes_per_cluster = _spikes_per_cluster(new_spike_clusters)
        self._spikes_per_cluster.update(new_spikes_per_cluster)
        # All old clusters are deleted.
        for cluster in _unique(old_spike_clusters):
            del self._spikes_per_cluster[cluster]

        # We return the UpdateInfo structure.
        up = _assign_update_info(spike_ids,
                                 old_spike_clusters,
                                 new_spike_clusters)

        # We make the assignements.
        self._spike_clusters[spike_ids] = new_spike_clusters

        return up

    def assign(self, spike_ids, spike_clusters_rel=0):
        """Assign clusters to a number of spikes.

        NOTE: spike_clusters_rel contains relative indices. They don't
        correspond to final cluster ids: self.assign() handles the final
        assignements to ensure that no cluster ends up modified. A cluster
        can only be born, stay unchanged, or die.

        """

        assert not isinstance(spike_ids, slice)

        # Ensure 'spike_clusters_rel' is an array-like.
        if not hasattr(spike_clusters_rel, '__len__'):
            spike_clusters_rel = spike_clusters_rel * np.ones(len(spike_ids),
                                                              dtype=np.int64)

        assert len(spike_ids) == len(spike_clusters_rel)

        # Normalize the spike-cluster assignement such that
        # there are only new or dead clusters, not modified clusters.
        # This implies that spikes not explicitely selected, but that
        # belong to clusters affected by the operation, will be assigned
        # to brand new clusters.
        spike_ids, cluster_ids = _extend_assignement(spike_ids,
                                                     self._spike_clusters,
                                                     spike_clusters_rel)

        up = self._do_assign(spike_ids, cluster_ids)

        # Add the assignement to the undo stack.
        self._undo_stack.add((spike_ids, cluster_ids))

        return up

    def split(self, spike_ids):
        """Split a number of spikes into a new cluster."""
        # self.assign() accepts relative numbers as second argument.
        return self.assign(spike_ids, 0)

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

        # What are the spikes affected by the last changes?
        changed = np.nonzero(self._spike_clusters !=
                             spike_clusters_new)[0]
        clusters_changed = spike_clusters_new[changed]

        return self._do_assign(changed,
                               # self._spike_clusters[changed],
                               clusters_changed)

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
        return self._do_assign(spike_ids,
                               # self._spike_clusters[spike_ids],
                               cluster_ids)
