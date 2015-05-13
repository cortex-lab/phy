# -*- coding: utf-8 -*-

"""Clustering utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from copy import deepcopy

import numpy as np

from ._history import History
from ...utils import _index_of, Bunch, _as_list, _as_array


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _spikes_in_clusters(spike_clusters, clusters):
    """Return the ids of all spikes belonging to the specified clusters."""
    if len(spike_clusters) == 0 or len(clusters) == 0:
        return np.array([], dtype=np.int)
    return np.nonzero(np.in1d(spike_clusters, clusters))[0]


def _spikes_per_cluster(spike_ids, spike_clusters):
    """Return a dictionary {cluster: list_of_spikes}."""
    rel_spikes = np.argsort(spike_clusters)
    abs_spikes = spike_ids[rel_spikes]
    spike_clusters = spike_clusters[rel_spikes]

    diff = np.empty_like(spike_clusters)
    diff[0] = 1
    diff[1:] = np.diff(spike_clusters)

    idx = np.nonzero(diff > 0)[0]
    clusters = spike_clusters[idx]

    spikes_in_clusters = {clusters[i]: np.sort(abs_spikes[idx[i]:idx[i+1]])
                          for i in range(len(clusters) - 1)}
    spikes_in_clusters[clusters[-1]] = np.sort(abs_spikes[idx[-1]:])

    return spikes_in_clusters


def _flatten_spikes_per_cluster(spikes_per_cluster):
    """Convert a dictionary {cluster: list_of_spikes} to a
    spike_clusters array."""
    clusters = sorted(spikes_per_cluster)
    clusters_arr = np.concatenate([(cluster *
                                   np.ones(len(spikes_per_cluster[cluster])))
                                   for cluster in clusters]).astype(np.int64)
    spikes_arr = np.concatenate([spikes_per_cluster[cluster]
                                 for cluster in clusters])
    spike_clusters = np.vstack((spikes_arr, clusters_arr))
    ind = np.argsort(spike_clusters[0, :])
    return spike_clusters[1, ind]


def _concatenate_per_cluster_arrays(spikes_per_cluster, arrays):
    """Concatenate arrays from a {cluster: array} dictionary."""
    assert set(arrays) <= set(spikes_per_cluster)
    clusters = sorted(arrays)
    # Check the sizes of the spikes per cluster and the arrays.
    n_0 = [len(spikes_per_cluster[cluster]) for cluster in clusters]
    n_1 = [len(arrays[cluster]) for cluster in clusters]
    assert n_0 == n_1

    # Concatenate all spikes to find the right insertion order.
    spikes = np.concatenate([spikes_per_cluster[cluster]
                             for cluster in clusters])
    idx = np.argsort(spikes)
    # NOTE: concatenate all arrays along the first axis, because we assume
    # that the first axis represents the spikes.
    arrays = np.concatenate([_as_array(arrays[cluster])
                             for cluster in clusters])
    return arrays[idx, ...]


def _subset_spikes_per_cluster(spikes_per_cluster, arrays, spikes_sub,
                               allow_cut=False):
    """Cut spikes_per_cluster and arrays along a list of spikes."""
    # WARNING: spikes_sub should be sorted and without duplicates.
    spikes_sub = _as_array(spikes_sub)
    spikes_per_cluster_subset = {}
    arrays_subset = {}
    n = 0

    # Opt-in parameter to allow cutting the requested spikes with
    # the spikes per cluster dictionary.
    _all_spikes = np.hstack(spikes_per_cluster.values())
    if allow_cut:
        spikes_sub = np.intersect1d(spikes_sub,
                                    _all_spikes)
    assert np.all(np.in1d(spikes_sub, _all_spikes))

    for cluster in sorted(spikes_per_cluster):
        spikes_c = _as_array(spikes_per_cluster[cluster])
        array = _as_array(arrays[cluster])
        assert spikes_sub.dtype == spikes_c.dtype
        spikes_sc = np.intersect1d(spikes_sub, spikes_c)
        # assert spikes_sc.dtype == np.int64
        spikes_per_cluster_subset[cluster] = spikes_sc
        idx = _index_of(spikes_sc, spikes_c)
        arrays_subset[cluster] = array[idx, ...]
        assert len(spikes_sc) == len(arrays_subset[cluster])
        n += len(spikes_sc)
    assert n == len(spikes_sub)
    return spikes_per_cluster_subset, arrays_subset


def _update_cluster_selection(clusters, up):
    clusters = list(clusters)
    # Remove deleted clusters.
    clusters = [clu for clu in clusters if clu not in up.deleted]
    # Add new clusters at the end of the selection.
    return clusters + [clu for clu in up.added if clu not in clusters]


#------------------------------------------------------------------------------
# UpdateInfo class
#------------------------------------------------------------------------------

def update_info(**kwargs):
    """Hold information about clustering changes."""
    d = dict(
        description=None,  # information about the update: 'merge', 'assign',
                           # or 'metadata_<name>'
        history=None,  # None, 'undo', or 'redo'
        spikes=[],  # all spikes affected by the update
        added=[],  # new clusters
        deleted=[],  # deleted clusters
        descendants=[],  # pairs of (old_cluster, new_cluster)
        metadata_changed=[],  # clusters with changed metadata
        metadata_value=None,  # new metadata value
        old_spikes_per_cluster={},  # only for the affected clusters
        new_spikes_per_cluster={},  # only for the affected clusters
    )
    d.update(kwargs)
    return Bunch(d)


UpdateInfo = update_info


#------------------------------------------------------------------------------
# ClusterMetadataUpdater class
#------------------------------------------------------------------------------

class ClusterMetadataUpdater(object):
    """Handle cluster metadata changes."""
    def __init__(self, cluster_metadata):
        self._cluster_metadata = cluster_metadata
        # Keep a deep copy of the original structure for the undo stack.
        self._data_base = deepcopy(cluster_metadata.data)
        # The stack contains (clusters, field, value, update_info) tuples.
        self._undo_stack = History((None, None, None, None))

        for field, func in self._cluster_metadata._fields.items():

            # Create self.<field>(clusters).
            def _make_get(field):
                def f(clusters):
                    return self._cluster_metadata._get(clusters, field)
                return f
            setattr(self, field, _make_get(field))

            # Create self.set_<field>(clusters, value).
            def _make_set(field):
                def f(clusters, value):
                    return self._set(clusters, field, value)
                return f
            setattr(self, 'set_{0:s}'.format(field), _make_set(field))

    def _set(self, clusters, field, value, add_to_stack=True):
        self._cluster_metadata._set(clusters, field, value)
        clusters = _as_list(clusters)
        info = UpdateInfo(description='metadata_' + field,
                          metadata_changed=clusters,
                          metadata_value=value,
                          )
        if add_to_stack:
            self._undo_stack.add((clusters, field, value, info))
        return info

    def undo(self):
        """Undo the last metadata change.

        Returns
        -------

        up : UpdateInfo instance

        """
        args = self._undo_stack.back()
        if args is None:
            return
        self._cluster_metadata._data = deepcopy(self._data_base)
        for clusters, field, value, _ in self._undo_stack:
            if clusters is not None:
                self._set(clusters, field, value, add_to_stack=False)
        # Return the UpdateInfo instance of the undo action.
        info = args[-1]
        info.history = 'undo'
        return info

    def redo(self):
        """Redo the next metadata change.

        Returns
        -------

        up : UpdateInfo instance
        """
        args = self._undo_stack.forward()
        if args is None:
            return
        clusters, field, value, info = args
        self._set(clusters, field, value, add_to_stack=False)
        # Return the UpdateInfo instance of the redo action.
        info.history = 'redo'
        return info
