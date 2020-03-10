# -*- coding: utf-8 -*-

"""Probe view."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
import logging

import numpy as np

from phy.utils.color import selected_cluster_color
from phylib.utils.geometry import get_non_overlapping_boxes
from phy.plot.visuals import ScatterVisual, TextVisual
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Probe view
# -----------------------------------------------------------------------------

def _get_pos_data_bounds(positions):
    positions, _ = get_non_overlapping_boxes(positions)
    x, y = positions.T
    xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
    w = xmax - xmin
    h = ymax - ymin
    k = .05
    data_bounds = (xmin - w * k, ymin - h * k, xmax + w * k, ymax + h * k)
    return positions, data_bounds


class ProbeView(ManualClusteringView):
    """This view displays the positions of all channels on the probe, highlighting channels
    where the selected clusters belong.

    Constructor
    -----------

    positions : array-like
        An `(n_channels, 2)` array with the channel positions
    best_channels : function
        Maps `cluster_id` to the list of the best_channel_ids.
    channel_labels : list
        List of channel label strings.
    dead_channels : list
        List of dead channel ids.

    """

    # Do not show too many clusters.
    max_n_clusters = 20

    _default_position = 'right'

    # Marker size of channels without selected clusters.
    unselected_marker_size = 10

    # Marker size of channels with selected clusters.
    selected_marker_size = 15

    # Alpha value of the dead channels.
    dead_channel_alpha = .25

    do_show_labels = False

    def __init__(
            self, positions=None, best_channels=None, channel_labels=None,
            dead_channels=None, **kwargs):
        super(ProbeView, self).__init__(**kwargs)
        self.state_attrs += ('do_show_labels',)

        # Normalize positions.
        assert positions.ndim == 2
        assert positions.shape[1] == 2
        positions = positions.astype(np.float32)
        self.positions, self.data_bounds = _get_pos_data_bounds(positions)

        self.n_channels = positions.shape[0]
        self.best_channels = best_channels

        self.channel_labels = channel_labels or [str(ch) for ch in range(self.n_channels)]
        self.dead_channels = dead_channels if dead_channels is not None else ()

        self.probe_visual = ScatterVisual()
        self.canvas.add_visual(self.probe_visual)

        # Probe visual.
        color = np.ones((self.n_channels, 4))
        color[:, :3] = .5
        # Change alpha value for dead channels.
        if len(self.dead_channels):
            color[self.dead_channels, 3] = self.dead_channel_alpha
        self.probe_visual.set_data(
            pos=self.positions, data_bounds=self.data_bounds,
            color=color, size=self.unselected_marker_size)

        # Cluster visual.
        self.cluster_visual = ScatterVisual()
        self.canvas.add_visual(self.cluster_visual)

        # Text visual
        color[:] = 1
        color[self.dead_channels, :3] = self.dead_channel_alpha * 2
        self.text_visual = TextVisual()
        self.text_visual.inserter.insert_vert('uniform float n_channels;', 'header')
        self.text_visual.inserter.add_varying(
            'float', 'v_discard',
            'float((n_channels >= 200 * u_zoom.y) && '
            '(mod(int(a_string_index), int(n_channels / (200 * u_zoom.y))) >= 1))')
        self.text_visual.inserter.insert_frag('if (v_discard > 0) discard;', 'end')
        self.canvas.add_visual(self.text_visual)
        self.text_visual.set_data(
            pos=self.positions, text=self.channel_labels, anchor=[0, -1],
            data_bounds=self.data_bounds, color=color
        )
        self.text_visual.program['n_channels'] = self.n_channels
        self.canvas.update()

    def _get_clu_positions(self, cluster_ids):
        """Get the positions of the channels containing selected clusters."""

        # List of channels per cluster.
        cluster_channels = {i: self.best_channels(cl) for i, cl in enumerate(cluster_ids)}

        # List of clusters per channel.
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
                alpha = 1.0 if channel_id not in self.dead_channels else self.dead_channel_alpha
                clu_pos.append((x, y))
                clu_colors.append(selected_cluster_color(clu_idx, alpha=alpha))
        return np.array(clu_pos), np.array(clu_colors)

    def on_select(self, cluster_ids=(), **kwargs):
        """Update the view with the selected clusters."""
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return
        pos, colors = self._get_clu_positions(cluster_ids)
        self.cluster_visual.set_data(
            pos=pos, color=colors, size=self.selected_marker_size, data_bounds=self.data_bounds)
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(ProbeView, self).attach(gui)
        self.actions.add(self.toggle_show_labels, checkable=True, checked=self.do_show_labels)

        if not self.do_show_labels:
            self.text_visual.hide()

    def toggle_show_labels(self, checked):
        """Toggle the display of the channel ids."""
        logger.debug("Set show labels to %s.", checked)
        self.do_show_labels = checked
        self.text_visual._hidden = not checked
        self.canvas.update()
