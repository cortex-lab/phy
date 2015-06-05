# -*- coding: utf-8 -*-

"""Clustering utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from copy import deepcopy

from ._history import History
from ...utils import Bunch, _as_list


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
