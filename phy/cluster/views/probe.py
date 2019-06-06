# -*- coding: utf-8 -*-

"""Probe view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
import logging

import numpy as np

from phylib.utils.color import selected_cluster_color
from phylib.utils.geometry import _get_boxes
from phy.plot.visuals import ScatterVisual
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Probe view
# -----------------------------------------------------------------------------

def _get_pos_data_bounds(positions):

    boxes = _get_boxes(positions, keep_aspect_ratio=False)
    xmin, ymin = boxes[:, :2].min(axis=0)
    xmax, ymax = boxes[:, 2:].max(axis=0)

    x = boxes[:, [0, 2]].mean(axis=1)
    y = boxes[:, [1, 3]].mean(axis=1)
    positions = np.c_[x, y]

    # x, y = positions.T
    # xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    w = xmax - xmin
    h = ymax - ymin
    k = .05
    data_bounds = (xmin - w * k, ymin - h * k, xmax + w * k, ymax + h * k)
    return positions, data_bounds


class ProbeView(ManualClusteringView):
    _default_position = 'right'
    unselected_marker_size = 10
    selected_marker_size = 15

    def __init__(self, positions=None, best_channels=None):
        super(ProbeView, self).__init__()

        # Normalize positions.
        assert positions.ndim == 2
        assert positions.shape[1] == 2
        self.positions, self.data_bounds = _get_pos_data_bounds(positions)

        self.n_channels = positions.shape[0]
        self.best_channels = best_channels

        self.probe_visual = ScatterVisual()
        self.canvas.add_visual(self.probe_visual)

        # Probe visual.
        self.probe_visual.set_data(
            pos=self.positions, data_bounds=self.data_bounds,
            color=(.5, .5, .5, 1.), size=self.unselected_marker_size)

        # Cluster visual.
        self.cluster_visual = ScatterVisual()
        self.canvas.add_visual(self.cluster_visual)

    def _get_clu_positions(self, cluster_ids):
        cluster_channels = {i: self.best_channels(cl) for i, cl in enumerate(cluster_ids)}

        clusters_per_channel = defaultdict(lambda: [])
        for clu_idx, channels in cluster_channels.items():
            for channel in channels:
                clusters_per_channel[channel].append(clu_idx)

        # Enumerate the discs for each channel.
        w = self.data_bounds[2] - self.data_bounds[0]
        clu_pos = []
        clu_colors = []
        for channel_id, (x, y) in enumerate(self.positions):
            for i, clu_idx in enumerate(clusters_per_channel[channel_id]):
                n = len(clusters_per_channel[channel_id])
                # Translation.
                t = .025 * w * (i - .5 * (n - 1))
                x += t
                clu_pos.append((x, y))
                clu_colors.append(selected_cluster_color(clu_idx))
        return np.array(clu_pos), np.array(clu_colors)

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return
        pos, colors = self._get_clu_positions(cluster_ids)
        self.cluster_visual.set_data(
            pos=pos, color=colors, size=self.selected_marker_size, data_bounds=self.data_bounds)
        self.canvas.update()
