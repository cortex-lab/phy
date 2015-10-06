# -*- coding: utf-8 -*-

"""Manual clustering GUI plugins."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from ._utils import ClusterMeta
from .clustering import Clustering
from .wizard import Wizard
from phy.gui.actions import Actions, Snippets
from phy.io.array import select_spikes
from phy.utils.plugin import IPlugin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Clustering objects
# -----------------------------------------------------------------------------

def create_cluster_meta(data):
    """Return a ClusterMeta instance with cluster group support."""
    meta = ClusterMeta()

    def group(ascendant_values=None):
        if not ascendant_values:  # pragma: no cover
            return 3
        s = list(set(ascendant_values) - set([None, 3]))
        # Return the default value if all ascendant values are the default.
        if not s:  # pragma: no cover
            return 3
        # Otherwise, return good (2) if it is present, or the largest value
        # among those present.
        return max(s)

    meta.add_field('group', 3, group)

    meta.from_dict(data)

    return meta


# -----------------------------------------------------------------------------
# Clustering GUI plugins
# -----------------------------------------------------------------------------

class ManualClustering(IPlugin):
    """Plugin that brings manual clustering facilities to a GUI:

    * Clustering instance: merge, split, undo, redo
    * ClusterMeta instance: change cluster metadata (e.g. group)
    * Wizard
    * Selection
    * Many manual clustering-related actions, snippets, shortcuts, etc.

    Bring the `select` event to the GUI. This is raised when clusters are
    selected by the user or by the wizard.

    Other plugins can connect to that event.

    """
    def attach_to_gui(self, gui,
                      spike_clusters=None,
                      cluster_meta=None,
                      n_spikes_max_per_cluster=100,
                      ):
        self.gui = gui

        # Create Clustering and ClusterMeta.
        self.clustering = Clustering(spike_clusters)
        self.cluster_meta = cluster_meta

        # Create the wizard and attach it to Clustering/ClusterMeta.
        self.wizard = Wizard()
        self.wizard.attach(self.clustering, cluster_meta)

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
            # Selection.
            actions.add(callback=self.select, alias='c')

            # Wizard.
            actions.add(callback=self.wizard.start, name='reset_wizard')
            actions.add(callback=self.wizard.first)
            actions.add(callback=self.wizard.last)
            actions.add(callback=self.wizard.previous)
            actions.add(callback=self.wizard.next)
            actions.add(callback=self.wizard.pin)
            actions.add(callback=self.wizard.unpin)

            # Clustering.
            actions.add(callback=self.merge)
            actions.add(callback=self.split)
            actions.add(callback=self.move)
            actions.add(callback=self.undo)
            actions.add(callback=self.redo)

        # Attach the GUI and register the actions.
        snippets.attach(gui, actions)
        actions.attach(gui)
        actions.reset()

        self.actions = actions
        self.snippets = snippets

    # Wizard-related actions
    # -------------------------------------------------------------------------

    def select(self, cluster_ids):
        self.wizard.selection = cluster_ids

    # Clustering actions
    # -------------------------------------------------------------------------

    def merge(self, cluster_ids=None):
        if cluster_ids is None:
            cluster_ids = self.wizard.selection
        self.clustering.merge(cluster_ids)

    def split(self, spike_ids):
        # TODO: connect to request_split emitted by view
        self.clustering.split(spike_ids)

    def move(self, clusters, group):
        # TODO
        pass

    def undo(self):
        self.clustering.undo()

    def redo(self):
        self.clustering.redo()
