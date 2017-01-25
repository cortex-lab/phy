# -*- coding: utf-8 -*-

"""Probe view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from phy.electrode.layout import probe_layout
from phy.gui import HTMLWidget

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Correlogram view
# -----------------------------------------------------------------------------

class ProbeView(HTMLWidget):
    def __init__(self, positions=None, best_channels=None):
        super(ProbeView, self).__init__()
        self.positions = positions
        self.best_channels = best_channels

    def on_select(self, cluster_ids, **kwargs):
        if not len(cluster_ids):
            return
        # TODO: consider all clusters with colors.
        channel_ids = self.best_channels(cluster_ids[0])
        self.set_body(probe_layout(self.positions, channel_ids))
        self.rebuild()

    def attach(self, gui):
        gui.connect_(self.on_select)
        self.show()
        gui.add_view(self)
