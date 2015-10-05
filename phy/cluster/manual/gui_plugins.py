# -*- coding: utf-8 -*-

"""Manual clustering GUI plugins."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from ._utils import ClusterMetadataUpdater
from .clustering import Clustering
from .wizard import Wizard
from phy.gui.actions import Actions, Snippets
from phy.io.array import select_spikes
from phy.utils.plugin import IPlugin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Clustering GUI plugins
# -----------------------------------------------------------------------------

class ManualClustering(IPlugin):
    def attach_to_gui(self, gui,
                      spike_clusters=None,
                      cluster_metadata=None,
                      n_spikes_max_per_cluster=100,
                      ):
        # Create Clustering and ClusterMetadataUpdater.
        clustering = Clustering(spike_clusters)
        cluster_meta_up = ClusterMetadataUpdater(cluster_metadata)

        # Create the wizard and attach it to Clustering/ClusterMetadataUpdater.
        wizard = Wizard()
        wizard.attach(clustering, cluster_meta_up)

        @wizard.connect
        def on_select(cluster_ids):
            """When the wizard selects clusters, choose a spikes subset
            and emit the `select` event on the GUI.

            The wizard is responsible for the notion of "selected clusters".

            """
            spike_ids = select_spikes(cluster_ids,
                                      n_spikes_max_per_cluster,
                                      clustering.spikes_per_cluster)
            gui.emit('select', cluster_ids, spike_ids)

        self.create_actions(gui)

    def create_actions(self, gui):
        actions = Actions()
        snippets = Snippets()

        # Create the default actions for the clustering GUI.
        @actions.connect
        def on_reset():
            actions.add(alias='s', callback=self.select)
            # TODO: other actions

        # Attach the GUI and register the actions.
        snippets.attach(gui, actions)
        actions.attach(gui)
        actions.reset()

    def toggle_correlogram_normalization(self):
        pass

    def toggle_waveforms_mean(self):
        pass

    def toggle_waveforms_overlap(self):
        pass

    def show_features_time(self):
        pass

    def select(self, cluster_ids):
        pass

    def reset_wizard(self):
        pass

    def first(self):
        pass

    def last(self):
        pass

    def next(self):
        pass

    def previous(self):
        pass

    def pin(self):
        pass

    def unpin(self):
        pass

    def merge(self, cluster_ids=None):
        pass

    def split(self, spike_ids=None):
        pass

    def move(self, clusters, group):
        pass

    def undo(self):
        pass

    def redo(self):
        pass
