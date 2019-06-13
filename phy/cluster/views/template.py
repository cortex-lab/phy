# -*- coding: utf-8 -*-

"""Template view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phylib.utils.color import _add_selected_clusters_colors
from phylib.io.array import _index_of
from phylib.utils import emit, Bunch
from phy.plot import _get_linear_x
from phy.plot.visuals import PlotVisual
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------

class TemplateView(ManualClusteringView):
    """This view shows all template waveforms of all clusters in a large grid of shape
    `(n_channels, n_clusters)`.

    Constructor:

    - `templates`: a function `cluster_ids => [Bunch(template, channel_ids)]` where `template` is
      an `(n_samples, n_channels)` array, and `channel_ids` represents the channel ids of the
      channels in the template array.

    - `channel_ids`: the list of all channel ids
    - `cluster_ids`: the list of all cluster ids
    - `cluster_color_selector`: the object responsible for the color mapping

    """
    _default_position = 'right'
    cluster_ids = ()
    _scaling = 1.
    _scaling_increment = 1.1

    default_shortcuts = {
        'increase': 'ctrl+alt++',
        'decrease': 'ctrl+alt+-',
    }

    def __init__(self, templates=None, channel_ids=None, cluster_ids=None,
                 cluster_color_selector=None):
        super(TemplateView, self).__init__()
        self.state_attrs += ('scaling',)
        self.local_state_attrs += ('scaling',)

        self.cluster_color_selector = cluster_color_selector
        # Full list of channels.
        self.channel_ids = channel_ids
        # Full list of clusters.
        self.set_cluster_ids(cluster_ids)
        # Total number of channels.
        self.n_channels = len(channel_ids)
        self.canvas.set_layout('grid', box_bounds=[[-1, -1, +1, +1]], has_clip=False)
        self.canvas.enable_axes()
        self.templates = templates
        self.visual = PlotVisual()
        self.canvas.add_visual(self.visual)
        self._cluster_box_index = {}

    # Internal plot functions
    # -------------------------------------------------------------------------

    def _get_data_bounds(self, bunchs):
        m = np.median([b.template.min() for b in bunchs.values()])
        M = np.median([b.template.max() for b in bunchs.values()])
        return [-1, m, +1, M]

    def _get_batch_data(self, bunch, cluster_id, cluster_idx):
        d = bunch
        wave = d.template  # shape: (n_samples, n_channels)
        channel_ids_loc = d.channel_ids
        n_channels_loc = len(channel_ids_loc)

        n_samples, nc = wave.shape
        assert nc == n_channels_loc

        # Find the x coordinates.
        t = _get_linear_x(n_channels_loc, n_samples)

        color = self.cluster_colors[cluster_idx]
        assert len(color) == 4

        # Generate the box index (channel_idx, cluster_idx) per vertex.
        box_index = _index_of(channel_ids_loc, self.channel_ids)
        box_index = np.repeat(box_index, n_samples)
        box_index = np.c_[
            box_index.reshape((-1, 1)), cluster_idx * np.ones((n_samples * n_channels_loc, 1))]
        assert box_index.shape == (n_channels_loc * n_samples, 2)
        assert box_index.size == wave.size * 2
        self._cluster_box_index[cluster_id] = box_index

        # Generate the waveform array.
        wave = wave.T * (self._scaling or 1.)

        return Bunch(
            x=t, y=wave, color=color, box_index=box_index)

    def _plot_templates(self, bunchs, data_bounds=None):
        if not bunchs:
            return
        self.visual.reset_batch()
        for i, cluster_id in enumerate(self.cluster_ids):
            data = self._get_batch_data(bunchs[cluster_id], cluster_id, i)
            self.visual.add_batch_data(**data, data_bounds=data_bounds)
        self.canvas.update_visual(self.visual)

    # Main methods
    # -------------------------------------------------------------------------

    def set_cluster_ids(self, cluster_ids):
        self.cluster_ids = cluster_ids
        self.cluster_colors = self.cluster_color_selector.get_colors(self.cluster_ids, alpha=.75)

    def update_cluster_sort(self, cluster_ids):
        """Update the order of all clusters."""

        # Only the order of the cluster_ids is supposed to change here.
        # We just have to update box_index instead of replotting everything.
        assert len(cluster_ids) == len(self.cluster_ids)
        self.cluster_ids = cluster_ids
        box_index = []
        for i, cluster_id in enumerate(self.cluster_ids):
            clu_box_index = self._cluster_box_index[cluster_id]
            clu_box_index[:, 1] = i
            box_index.append(clu_box_index)
        box_index = np.concatenate(box_index, axis=0)
        self.visual.set_box_index(box_index)
        self.canvas.update()

    def update_color(self, selected_clusters=None):
        """Update the color of the clusters, taking the selected clusters into account."""

        # This method is only used when the view has been plotted at least once,
        # such that self._cluster_box_index has been filled.
        if not self._cluster_box_index:
            self.plot()
        # The call to set_cluster_ids() update the cluster_colors array.
        self.set_cluster_ids(self.cluster_ids)
        # Selected cluster colors.
        cluster_colors = self.cluster_colors
        if selected_clusters is not None:
            cluster_colors = _add_selected_clusters_colors(
                selected_clusters, self.cluster_ids, cluster_colors)
        # Number of vertices per cluster = number of vertices per signal
        n_vertices_clu = [
            len(self._cluster_box_index[cluster_id]) for cluster_id in self.cluster_ids]
        # The argument passed to set_color() must have 1 row per vertex.
        self.visual.set_color(np.repeat(cluster_colors, n_vertices_clu, axis=0))
        self.canvas.update()

    def on_select(self, cluster_ids=(), **kwargs):
        if not cluster_ids:
            return
        self.update_color(selected_clusters=cluster_ids)

    def plot(self):
        """Make the template plot."""

        # Retrieve the waveform data.
        bunchs = self.templates(self.cluster_ids)
        n_clusters = len(self.cluster_ids)
        self.canvas.grid.shape = (self.n_channels, n_clusters)

        data_bounds = self._get_data_bounds(bunchs)
        self._plot_templates(bunchs, data_bounds=data_bounds)
        self.canvas.axes.reset_data_bounds((0, 0, n_clusters, self.n_channels), do_update=True)
        self.canvas.update()

    def on_mouse_click(self, e):
        """Select a cluster by clicking on its template waveform."""
        b = e.button
        if 'Control' in e.modifiers:
            # Get mouse position in NDC.
            (channel_idx, cluster_idx), _ = self.canvas.grid.box_map(e.pos)
            cluster_id = self.cluster_ids[cluster_idx]
            logger.debug("Click on cluster %d with button %s.", cluster_id, b)
            emit('cluster_click', self, cluster_id, button=b)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(TemplateView, self).attach(gui)
        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.separator()

    # Template scaling
    # -------------------------------------------------------------------------

    @property
    def scaling(self):
        """Scaling of the template waveforms."""
        return self._scaling or 1.

    @scaling.setter
    def scaling(self, value):
        self._scaling = value

    def increase(self):
        """Increase the scaling of the template waveforms."""
        self.scaling *= self._scaling_increment
        self.plot()

    def decrease(self):
        """Decrease the scaling of the template waveforms."""
        self.scaling /= self._scaling_increment
        self.plot()
