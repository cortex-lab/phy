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
from ...io.experiment import BaseExperiment
from ...plot.waveforms import WaveformView


#------------------------------------------------------------------------------
# Session class
#------------------------------------------------------------------------------

class Session(object):
    """Provide all user-exposed actions for a manual clustering session."""

    def __init__(self, experiment):
        if not isinstance(experiment, BaseExperiment):
            raise ValueError("'experiment' must be an instance of a "
                             "class deriving from BaseExperiment.")
        self._views = []
        self._global_history = GlobalHistory()
        self.experiment = experiment
        self._update()

    def select(self, clusters):
        """Select some clusters."""
        self.selector.selected_clusters = clusters
        self._update_views()

    def _update(self):
        """Initialize the Session after the channel group has changed."""
        # TODO: n_spikes_max
        spike_clusters = self.experiment.spike_clusters
        self.selector = Selector(spike_clusters, n_spikes_max=None)
        self.clustering = Clustering(spike_clusters)
        self.cluster_metadata = self.experiment.cluster_metadata
        # TODO: change channel group and change recording

    # Views.
    # -------------------------------------------------------------------------

    def register_view(self, view):
        """Register a view."""
        # Register the spike clusters.
        view.visual.spike_clusters = self.clustering.spike_clusters
        # Register the ClusterMetadata.
        view.visual.cluster_metadata = self.cluster_metadata
        self._views.append(view)
        self._update_view(view)

    def unregister_view(self, view):
        """Unregister a view."""
        # TODO: remove a view when it is closed.
        self._close_view(view)
        if view in self._views:
            self._views.remove(view)

    def _update_views(self):
        """Update all views after a selection change."""
        for view in self._views:
            self._update_view(view)

    def _update_view(self, view):
        """Update a view after a selection change."""
        if isinstance(view, WaveformView):
            self._update_waveform_view(view)
        view.update()

    def _update_waveform_view(self, view):
        """Update a WaveformView after a selection change."""
        spikes = self.selector.selected_spikes
        view.visual.waveforms = self.experiment.waveforms[spikes]
        view.visual.masks = self.experiment.masks[spikes]
        view.visual.spike_labels = spikes
        view.visual.channel_positions = self.experiment.probe.positions

    def _show_view(self, view):
        """Show a VisPy canvas view."""
        try:
            from vispy.app import Canvas
        except ImportError:
            raise ImportError("VisPy is required.")
        assert isinstance(view, Canvas)
        view.show()
        return view

    def _close_view(self, view):
        """Close a view."""
        view.close()

    def show_waveforms(self):
        """Show a new WaveformView."""
        view = WaveformView()
        self.register_view(view)
        return self._show_view(view)

    # Clustering actions.
    # -------------------------------------------------------------------------

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
