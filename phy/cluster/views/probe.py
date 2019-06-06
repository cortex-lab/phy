# -*- coding: utf-8 -*-

"""Probe view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from phy.gui import HTMLWidget
from phylib.electrode.layout import probe_layout
from phylib.utils import connect, unconnect

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Correlogram view
# -----------------------------------------------------------------------------

class ProbeView(HTMLWidget):
    def __init__(self, positions=None, best_channels=None):
        super(ProbeView, self).__init__()
        self.positions = positions
        self.best_channels = best_channels

    def on_select(self, cluster_ids=(), **kwargs):
        cluster_channels = {i: self.best_channels(cl) for i, cl in enumerate(cluster_ids)}
        self.builder.set_body(probe_layout(self.positions, cluster_channels))
        self.build()

    def attach(self, gui):
        self.on_select()
        self.show()
        gui.add_view(self, position='right')

        @connect
        def on_select(sender, cluster_ids):
            if sender.__class__.__name__ != 'Supervisor':
                return
            self.on_select(cluster_ids=cluster_ids)

        # Save the view state in the GUI state.
        @connect(sender=gui)
        def on_close(sender=None):
            unconnect(on_select)
