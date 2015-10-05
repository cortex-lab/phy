# -*- coding: utf-8 -*-

"""Manual clustering GUI plugins."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from ._utils import ClusterMetadata, ClusterMetadataUpdater
from .clustering import Clustering
from .wizard import Wizard
from phy.gui.actions import Actions, Snippets
from phy.io.array import select_spikes
from phy.utils.plugin import IPlugin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Clustering objects
# -----------------------------------------------------------------------------

def create_cluster_metadata(data):
    """Return a ClusterMetadata instance with cluster group support."""
    meta = ClusterMetadata(data=data)

    @meta.default
    def group(cluster, ascendant_values=None):
        if not ascendant_values:
            return 3
        s = list(set(ascendant_values) - set([None, 3]))
        # Return the default value if all ascendant values are the default.
        if not s:  # pragma: no cover
            return 3
        # Otherwise, return good (2) if it is present, or the largest value
        # among those present.
        return max(s)

    return meta


# -----------------------------------------------------------------------------
# Clustering GUI plugins
# -----------------------------------------------------------------------------

class ManualClustering(IPlugin):
    """Plugin that brings manual clustering facilities to a GUI:

    * Clustering instance: merge, split, undo, redo
    * ClusterMetadataUpdater instance: change cluster metadata (e.g. group)
    * Wizard
    * Selection
    * Many manual clustering-related actions, snippets, shortcuts, etc.

    Bring the `select` event to the GUI. This is raised when clusters are
    selected by the user or by the wizard.

    Other plugins can connect to that event.

    """
    def attach_to_gui(self, gui,
                      spike_clusters=None,
                      cluster_metadata=None,
                      n_spikes_max_per_cluster=100,
                      ):
        # Create Clustering and ClusterMetadataUpdater.
        self.clustering = Clustering(spike_clusters)
        self.cluster_metadata = cluster_metadata
        cluster_meta_up = ClusterMetadataUpdater(cluster_metadata)

        # Create the wizard and attach it to Clustering/ClusterMetadataUpdater.
        self.wizard = Wizard()
        self.wizard.attach(self.clustering, cluster_meta_up)

        @self.wizard.connect
        def on_select(cluster_ids):
            """When the wizard selects clusters, choose a spikes subset
            and emit the `select` event on the GUI.

            The wizard is responsible for the notion of "selected clusters".

            """
            spike_ids = select_spikes(cluster_ids,
                                      n_spikes_max_per_cluster,
                                      self.clustering.spikes_per_cluster)
            gui.emit('select', cluster_ids, spike_ids)

        self.create_actions(gui)

        return self

    @property
    def cluster_ids(self):
        return self.clustering.cluster_ids

    def create_actions(self, gui):
        actions = Actions()
        snippets = Snippets()

        # Create the default actions for the clustering GUI.
        @actions.connect
        def on_reset():
            actions.add(callback=self.select, alias='s')
            # TODO: other actions

        # Attach the GUI and register the actions.
        snippets.attach(gui, actions)
        actions.attach(gui)
        actions.reset()

    # Wizard-related actions
    # -------------------------------------------------------------------------

    def select(self, cluster_ids):
        self.wizard.selection = cluster_ids

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

    # Clustering actions
    # -------------------------------------------------------------------------

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

    # View-related actions
    # -------------------------------------------------------------------------

    def toggle_correlogram_normalization(self):
        pass

    def toggle_waveforms_mean(self):
        pass

    def toggle_waveforms_overlap(self):
        pass

    def show_features_time(self):
        pass
