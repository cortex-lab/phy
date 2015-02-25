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

def _extend_spikes(spike_clusters, spike_ids):
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


def _extend_assignement(old_spike_clusters, spike_ids, spike_clusters_rel):
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
    extended_spike_ids = _extend_spikes(old_spike_clusters, spike_ids)
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
    update_info = UpdateInfo(description='assign',
                             spikes=spike_ids,
                             added=list(new_clusters),
                             deleted=list(old_clusters))
    return update_info


class Clustering(object):
    """Object representing a mapping from spike to cluster ids."""

    def __init__(self, spike_clusters):
        self._undo_stack = History(base_item=(None, None))
        # Spike -> cluster mapping.
        self._spike_clusters = _as_array(spike_clusters)
        # Update the spikes per cluster structure.
        self._spikes_per_cluster = {}
        self._update_spikes_per_cluster()
        # Keep a copy of the original spike clusters assignement.
        self._spike_clusters_base = self._spike_clusters.copy()

    def _update_spikes_per_cluster(self, spike_ids=None, spike_clusters=None):
        """Update the spikes_per_cluster structure after an assign operation.

        WARNING: this is a potentially heavy operation.

        WARNING 2: this needs to be called *before* updating
                   self._spike_clusters.

        """
        if spike_ids is None:
            assert spike_clusters is None
            # Compute the structure for all clusters.
            self._spikes_per_cluster = _spikes_per_cluster(self.spike_clusters)
        else:
            # Compute the spikes per cluster for part of the clusters only.
            assert spike_clusters is not None
            # Ensure spike_clusters has the right shape.
            if len(spike_clusters) == 1 and len(spike_ids) > 1:
                spike_clusters = np.ones(len(spike_ids)) * spike_clusters[0]
            assert len(spike_ids) == len(spike_clusters)
            # Ensure we have arrays.
            spike_ids = _as_array(spike_ids)
            spike_clusters = _as_array(spike_clusters)
            # Find old and new spike clusters.
            new_spike_clusters = spike_clusters
            old_spike_clusters = self._spike_clusters[spike_ids]
            assert len(old_spike_clusters) == len(new_spike_clusters)

            # Contain the list of spikes per cluster for all modified clusters.
            new_dict = _spikes_per_cluster(new_spike_clusters)

            # WARNING: all clusters appearing in self.spike_clusters should
            # be new clusters, i.e. they should not appear in the current
            # self._spikes_per_cluster. This is because we assume that
            # a given cluster can never be modified: it can only die (and
            # a new cluster is created).
            new_clusters = set(new_dict)
            old_clusters = set(_unique(old_spike_clusters))
            assert not new_clusters.intersection(old_clusters)

            # We delete the old clusters
            for old_cluster in old_clusters:
                del self._spikes_per_cluster[old_cluster]

            # Finally, we update the structure with the new dictionary.
            self._spikes_per_cluster.update(new_dict)

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
        update_info = UpdateInfo(description='merge',
                                 spikes=spike_ids,
                                 added=[to],
                                 deleted=cluster_ids)

        # Update the spikes_per_cluster structure directly.
        self._spikes_per_cluster[to] = spike_ids
        for cluster in cluster_ids:
            del self._spikes_per_cluster[cluster]

        # Assign the clusters.
        self.spike_clusters[spike_ids] = to

        # Add to stack.
        self._undo_stack.add((spike_ids, [to]))

        return update_info

    def assign(self, spike_ids, spike_clusters_rel):
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

        if _is_array_like(spike_ids):
            assert len(spike_ids) == len(spike_clusters_rel)

        # Normalize the spike-cluster assignement such that
        # there are only new or dead clusters, not modified clusters.
        # This implies that spikes not explicitely selected, but that
        # belong to clusters affected by the operation, will be assigned
        # to brand new clusters.
        spike_ids, cluster_ids = _extend_assignement(self._spike_clusters,
                                                     spike_ids,
                                                     spike_clusters_rel)

        # Update the spikes per cluster structure.
        self._update_spikes_per_cluster(spike_ids, cluster_ids)

        # Make the assignements.
        self._spike_clusters[spike_ids] = cluster_ids

        # Add the assignement to the undo stack.
        assert len(spike_ids) == len(cluster_ids)
        self._undo_stack.add((spike_ids, cluster_ids))

        # We return the update info structure.
        return _assign_update_info(spike_ids,
                                   self._spike_clusters,
                                   cluster_ids)

    def split(self, spike_ids):
        """Split a number of spikes into a new cluster."""
        # self.assign() accepts relative numbers as second argument.
        return self.assign(spike_ids, self.new_cluster_id())

    def _do_assign(self, spikes, old_spike_clusters,
                   new_spike_clusters):
        # We return the UpdateInfo structure.
        up =  _assign_update_info(spikes,
                                  old_spike_clusters,
                                  new_spike_clusters)

        # WARNING: we make the assignements *after* creating the UpdateInfo
        # structure, because we used self._spike_clusters as the *old*
        # spike_clusters array.
        self._spike_clusters[spikes] = new_spike_clusters

        return up

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

        # Do the update.
        self._update_spikes_per_cluster(changed, clusters_changed)

        return self._do_assign(changed,
                               self._spike_clusters[changed],
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

        # Do the update.
        self._update_spikes_per_cluster(spike_ids, cluster_ids)

        # We apply the new assignement.
        return self._do_assign(spike_ids,
                               self._spike_clusters[spike_ids],
                               cluster_ids)
