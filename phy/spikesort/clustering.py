# -*- coding: utf-8 -*-

"""Clustering structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import namedtuple, defaultdict, OrderedDict

import numpy as np

from ..ext.six import iterkeys, itervalues
from ._utils import _unique, _spikes_in_clusters


#------------------------------------------------------------------------------
# History class
#------------------------------------------------------------------------------

class History(object):
    """Implement a history of actions with an undo stack."""
    def __init__(self, base_item=None):
        self.clear(base_item)

    def clear(self, base_item=None):
        """Clear the history."""
        # List of changes, contains at least the base item.
        self._history = [base_item]
        # Index of the current item.
        self._index = 0

    @property
    def current_item(self):
        """Return the current element."""
        if self._history and self._index >= 0:
            self._check_index()
            return self._history[self._index]

    @property
    def current_position(self):
        """Current position in the history."""
        return self._index

    def _check_index(self):
        """Check that the index is without the bounds of _history."""
        assert 0 <= self._index <= len(self._history) - 1
        # There should always be the base item at least.
        assert len(self._history) >= 1

    def iter(self, start=0, end=None):
        """Iterate through successive history items.

        Parameters
        ----------
        end : int
            Index of the last item to loop through + 1.

        start : int
            Initial index for the loop (0 by default).

        """
        if end is None:
            end = self._index + 1
        elif end == 0:
            raise StopIteration()
        if start >= end:
            raise StopIteration()
        # Check arguments.
        assert 0 <= end <= len(self._history)
        assert 0 <= start <= end - 1
        for i in range(start, end):
            yield self._history[i]

    def __iter__(self):
        return self.iter()

    def __len__(self):
        return len(self._history)

    def add(self, item):
        """Add an item in the history."""
        self._check_index()
        # Possibly truncate the history up to the current point.
        self._history = self._history[:self._index + 1]
        # Append the item
        self._history.append(item)
        # Increment the index.
        self._index += 1
        self._check_index()
        # Check that the current element is what was provided to the function.
        assert id(self.current_item) == id(item)

    def back(self):
        """Go back in history if possible.

        Return the current item after going back.

        """
        if self._index <= 0:
            return None
        self._index -= 1
        self._check_index()
        return self.current_item

    def forward(self):
        """Go forward in history if possible.

        Return the current item after going forward.

        """
        if self._index >= len(self._history) - 1:
            return None
        self._index += 1
        self._check_index()
        return self.current_item


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


#------------------------------------------------------------------------------
# ClusterManager class
#------------------------------------------------------------------------------

class ClusterManager(object):
    """Object holding cluster metadata.

    Constructor
    -----------

    fields : list
        List of tuples (field_name, default_value).

    """

    def __init__(self, fields):
        # 'fields' is a list of tuples (field_name, default_value).
        # 'self._fields' is an OrderedDict {field_name ==> default_value}.
        self._fields = OrderedDict(fields)
        self._field_names = list(iterkeys(self._fields))
        # '_data' maps cluster labels to dict (field => value).
        self._data = defaultdict(dict)
        self._spike_clusters = None
        self._cluster_labels = None

    @property
    def data(self):
        """Dictionary holding data for all clusters."""
        return self._data

    @property
    def spike_clusters(self):
        """Mapping spike ==> cluster label."""
        return self._spike_clusters

    @spike_clusters.setter
    def spike_clusters(self, value):
        self._spike_clusters = value
        self._update()

    @property
    def cluster_labels(self):
        """Sorted list of non-empty cluster labels."""
        return self._cluster_labels

    def delete_clusters(self, clusters):
        """Delete clusters from the structure."""
        for cluster in clusters:
            del self._data[cluster]

    def _update(self, cluster_labels=None):
        """Remove empty clusters in the structure.

        Parameters
        ----------

        cluster_labels : array_like
            Sorted list of all cluster labels. By default, the list of unique
            clusters appearing in `self.spike_clusters`.

        """
        if cluster_labels is None:
            cluster_labels = _unique(self._spike_clusters)
        # Update the list of unique cluster labels.
        self._cluster_labels = cluster_labels
        # Find the clusters in the structure but not in the up-to-date
        # list of clusters.
        to_delete = set(iterkeys(self._data)) - set(self._cluster_labels)
        # Delete these clusters.
        self.delete_clusters(to_delete)

    def set(self, cluster, field, value):
        """Set some information for a cluster."""
        self._data[cluster][field] = value

    def get(self, cluster, field):
        """Get a specific information for a given cluster."""
        if cluster in self._data:
            info = self._data[cluster]
            if field in info:
                # Return the value.
                return info[field]
            else:
                # Or return the default value.
                return self._fields[field]

    def __getitem__(self, key):
        """Return the field values of all clusters, or all information of a
        cluster."""
        if key in self._data:
            # 'key' is a cluster label.
            return OrderedDict((field, self.get(key, field))
                               for field in self._field_names)
        elif key in self._field_names:
            # 'key' is a field name.
            return OrderedDict((cluster, self.get(cluster, key))
                               for cluster in self.cluster_labels)
        else:
            raise ValueError("Key {0:s} not in the list ".format(key) +
                             "of fields {1:s}".format(str(self._field_names)))
