# -*- coding: utf-8 -*-

"""Session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ._history import GlobalHistory
from .clustering import Clustering
from .cluster_metadata import ClusterMetadata
from .selector import Selector


#------------------------------------------------------------------------------
# Session class
#------------------------------------------------------------------------------

class Session(object):
    """Provide all user-exposed actions for a manual clustering session."""

    def __init__(self, experiment, channel_group=0):
        self._global_history = GlobalHistory()
        self.experiment = experiment
        # TODO: n_spikes_max
        spike_clusters = experiment.spike_clusters(channel_group=channel_group)
        self.selector = Selector(spike_clusters, n_spikes_max=None)
        self.clustering = Clustering(spike_clusters)
        self.cluster_metadata = experiment.cluster_metadata()
        self._views = []

    def register_view(self, view):
        """Register a view so that it gets updated after clustering actions."""
        self._views.append(view)

    def select(self, clusters):
        self.selector.selected_clusters = clusters
        self._update_views()

    def _update_views(self):
        for view in self._views:
            self._update_view(view)

    def _update_view(self, view):
        # TODO
        pass

    def _clustering_updated(self, up):
        """Update the selectors and views with an UpdateInfo object."""

        # TODO: Update the similarity matrix.
        # stats.update(up)

        # This doesn't do anything yet.
        self.selector.update(up)

        # Update the views.
        [view.update(up) for view in self._views]

    def _assigned(self, up):
        """Called when cluster assignements has been changed."""

        # Update the cluster metadata.
        self.cluster_metadata.update(up)

        # Save the action in the global stack.
        # TODO: add self.cluster_metadata once clustering actions involve
        # changes in cluster metadata.
        self._global_history.action(self.clustering)

        # Update all structures.
        self._clustering_updated(up)

    def merge(self, clusters):
        """Merge clusters."""
        up = self.clustering.merge(clusters)
        self._assigned(up)

    def split(self, spikes):
        """Create a new cluster from a selection of spikes."""
        up = self.clustering.split(spikes)
        self._assigned(up)

    def move(self, clusters, group):
        """Move clusters to a group."""
        self.cluster_metadata.set(clusters, 'group', group)
        self._global_history.action(self.cluster_metadata)

    def undo(self):
        """Undo the last action."""
        up = self._global_history.undo()
        self._clustering_updated(up)

    def redo(self):
        """Redo the last undone action."""
        up = self._global_history.redo()
        self._clustering_updated(up)

    def wizard_start(self):
        raise NotImplementedError()

    def wizard_next(self):
        raise NotImplementedError()

    def wizard_previous(self):
        raise NotImplementedError()

    def wizard_reset(self):
        raise NotImplementedError()
