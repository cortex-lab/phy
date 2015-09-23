# -*- coding: utf-8 -*-

"""Clustering utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from copy import deepcopy
from collections import defaultdict

from ._history import History
from phy.utils import Bunch, _as_list, _is_list


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _update_cluster_selection(clusters, up):
    clusters = list(clusters)
    # Remove deleted clusters.
    clusters = [clu for clu in clusters if clu not in up.deleted]
    # Add new clusters at the end of the selection.
    return clusters + [clu for clu in up.added if clu not in clusters]


def _join(clusters):
    return '[{}]'.format(', '.join(map(str, clusters)))


#------------------------------------------------------------------------------
# UpdateInfo class
#------------------------------------------------------------------------------

class UpdateInfo(Bunch):
    """Hold information about clustering changes."""
    def __init__(self, **kwargs):
        d = dict(
            description='',  # information about the update: 'merge', 'assign',
                             # or 'metadata_<name>'
            history=None,  # None, 'undo', or 'redo'
            spike_ids=[],  # all spikes affected by the update
            added=[],  # new clusters
            deleted=[],  # deleted clusters
            descendants=[],  # pairs of (old_cluster, new_cluster)
            metadata_changed=[],  # clusters with changed metadata
            metadata_value=None,  # new metadata value
            old_spikes_per_cluster={},  # only for the affected clusters
            new_spikes_per_cluster={},  # only for the affected clusters
            selection=[],  # clusters selected before the action
        )
        d.update(kwargs)
        super(UpdateInfo, self).__init__(d)

    def __repr__(self):
        desc = self.description
        h = ' ({})'.format(self.history) if self.history else ''
        if not desc:
            return '<UpdateInfo>'
        elif desc in ('merge', 'assign'):
            a, d = _join(self.added), _join(self.deleted)
            return '<{desc}{h} {d} => {a}>'.format(desc=desc,
                                                   a=a,
                                                   d=d,
                                                   h=h,
                                                   )
        elif desc.startswith('metadata'):
            c = _join(self.metadata_changed)
            m = self.metadata_value
            return '<{desc}{h} {c} => {m}>'.format(desc=desc,
                                                   c=c,
                                                   m=m,
                                                   h=h,
                                                   )
        return '<UpdateInfo>'


#------------------------------------------------------------------------------
# ClusterMetadataUpdater class
#------------------------------------------------------------------------------

class ClusterMetadata(object):
    """Hold cluster metadata.

    Features
    --------

    * New metadata fields can be easily registered
    * Arbitrary functions can be used for default values

    Notes
    ----

    If a metadata field `group` is registered, then two methods are
    dynamically created:

    * `group(cluster)` returns the group of a cluster, or the default value
      if the cluster doesn't exist.
    * `set_group(cluster, value)` sets a value for the `group` metadata field.

    """
    def __init__(self, data=None):
        self._fields = {}
        self._data = defaultdict(dict)
        # Fill the existing values.
        if data is not None:
            self._data.update(data)

    @property
    def data(self):
        return self._data

    def _get_one(self, cluster, field):
        """Return the field value for a cluster, or the default value if it
        doesn't exist."""
        if cluster in self._data:
            if field in self._data[cluster]:
                return self._data[cluster][field]
            elif field in self._fields:
                # Call the default field function.
                return self._fields[field](cluster)
        else:
            if field in self._fields:
                return self._fields[field](cluster)

    def _get(self, clusters, field):
        if _is_list(clusters):
            return [self._get_one(cluster, field)
                    for cluster in _as_list(clusters)]
        else:
            return self._get_one(clusters, field)

    def _set_one(self, cluster, field, value):
        """Set a field value for a cluster."""
        self._data[cluster][field] = value

    def _set(self, clusters, field, value):
        clusters = _as_list(clusters)
        for cluster in clusters:
            self._set_one(cluster, field, value)

    def default(self, func):
        """Register a new metadata field with a function
        returning the default value of a cluster."""
        field = func.__name__
        # Register the decorated function as the default field function.
        self._fields[field] = func
        # Create self.<field>(clusters).
        setattr(self, field, lambda clusters: self._get(clusters, field))
        # Create self.set_<field>(clusters, value).
        setattr(self, 'set_{0:s}'.format(field),
                lambda clusters, value: self._set(clusters, field, value))
        return func


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
