# -*- coding: utf-8 -*-

"""Feature view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging
import re

import numpy as np

from phylib.utils import Bunch, emit
from phy.utils.color import selected_cluster_color
from phy.plot.transform import Range
from phy.plot.visuals import ScatterVisual, TextVisual, LineVisual
from .base import ManualClusteringView, MarkerSizeMixin, ScalingMixin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Feature view
# -----------------------------------------------------------------------------

def _get_default_grid():
    """In the grid specification, 0 corresponds to the best channel, 1
    to the second best, and so on. A, B, C refer to the PC components."""
    s = """
    time,0A 1A,0A   0B,0A   1B,0A
    0A,1A   time,1A 0B,1A   1B,1A
    0A,0B   1A,0B   time,0B 1B,0B
    0A,1B   1A,1B   0B,1B   time,1B
    """.strip()
    return [[_ for _ in re.split(' +', line.strip())] for line in s.splitlines()]


def _get_point_color(clu_idx=None):
    if clu_idx is not None:
        color = selected_cluster_color(clu_idx, .5)
    else:
        color = (.5,) * 4
    assert len(color) == 4
    return color


def _get_point_masks(masks=None, clu_idx=None):
    masks = masks if masks is not None else 1.
    # NOTE: we add the cluster relative index for the computation of the depth on the GPU.
    return masks * .99999 + (clu_idx or 0)


def _get_masks_max(px, py):
    mx = px.get('masks', None)
    my = py.get('masks', None)
    if mx is None or my is None:
        return None
    return np.maximum(mx, my)


def _uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class FeatureView(MarkerSizeMixin, ScalingMixin, ManualClusteringView):
    """This view displays a 4x4 subplot matrix with different projections of the principal
    component features. This view keeps track of which channels are currently shown.

    Constructor
    -----------

    features : function
        Maps `(cluster_id, channel_ids=None, load_all=False)` to
        `Bunch(data, channel_ids, channel_labels, spike_ids , masks)`.
        * `data` is an `(n_spikes, n_channels, n_features)` array
        * `channel_ids` contains the channel ids of every row in `data`
        * `channel_labels` contains the channel labels of every row in `data`
        * `spike_ids` is a `(n_spikes,)` array
        * `masks` is an `(n_spikes, n_channels)` array

        This allows for a sparse format.

    attributes : dict
        Maps an attribute name to a 1D array with `n_spikes` numbers (for example, spike times).

    """

    # Do not show too many clusters.
    max_n_clusters = 8

    _default_position = 'right'
    cluster_ids = ()

    # Whether to disable automatic selection of channels.
    fixed_channels = False
    feature_scaling = 1.

    default_shortcuts = {
        'change_marker_size': 'alt+wheel',
        'increase': 'ctrl++',
        'decrease': 'ctrl+-',
        'add_lasso_point': 'ctrl+click',
        'stop_lasso': 'ctrl+right click',
        'toggle_automatic_channel_selection': 'c',
    }

    def __init__(self, features=None, attributes=None, **kwargs):
        super(FeatureView, self).__init__(**kwargs)
        self.state_attrs += ('fixed_channels', 'feature_scaling')

        assert features
        self.features = features
        self._lim = 1

        self.grid_dim = _get_default_grid()  # 2D array where every item a string like `0A,1B`
        self.n_rows, self.n_cols = np.array(self.grid_dim).shape
        self.canvas.set_layout('grid', shape=(self.n_rows, self.n_cols))
        self.canvas.enable_lasso()

        # Channels being shown.
        self.channel_ids = None

        # Attributes: extra features. This is a dictionary
        # {name: array}
        # where each array is a `(n_spikes,)` array.
        self.attributes = attributes or {}

        self.visual = ScatterVisual()
        self.canvas.add_visual(self.visual)

        self.text_visual = TextVisual()
        self.canvas.add_visual(self.text_visual)

        self.line_visual = LineVisual()
        self.canvas.add_visual(self.line_visual)

    def set_grid_dim(self, grid_dim):
        """Change the grid dim dynamically.

        Parameters
        ----------
        grid_dim : array-like (2D)
            `grid_dim[row, col]` is a string with two values separated by a comma. Each value
            is the relative channel id (0, 1, 2...) followed by the PC (A, B, C...). For example,
            `grid_dim[row, col] = 0B,1A`. Each value can also be an attribute name, for example
            `time`. For example, `grid_dim[row, col] = time,2C`.

        """
        self.grid_dim = grid_dim
        self.n_rows, self.n_cols = np.array(grid_dim).shape
        self.canvas.grid.shape = (self.n_rows, self.n_cols)

    # Internal methods
    # -------------------------------------------------------------------------

    def _iter_subplots(self):
        """Yield (i, j, dim)."""
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                dim = self.grid_dim[i][j]
                dim_x, dim_y = dim.split(',')
                yield i, j, dim_x, dim_y

    def _get_axis_label(self, dim):
        """Return the channel label from a dimension, if applicable."""
        if str(dim[:-1]).isdecimal():
            n = len(self.channel_ids)
            channel_id = self.channel_ids[int(dim[:-1]) % n]
            return self.channel_labels[channel_id] + dim[-1]
        else:
            return dim

    def _get_channel_and_pc(self, dim):
        """Return the channel_id and PC of a dim."""
        if self.channel_ids is None:
            return
        assert dim not in self.attributes  # This is called only on PC data.
        s = 'ABCDEFGHIJ'
        # Channel relative index, typically just 0 or 1.
        c_rel = int(dim[:-1])
        # Get the channel_id from the currently-selected channels.
        channel_id = self.channel_ids[c_rel % len(self.channel_ids)]
        pc = s.index(dim[-1])
        return channel_id, pc

    def _get_axis_data(self, bunch, dim, cluster_id=None, load_all=None):
        """Extract the points from the data on a given dimension.

        bunch is returned by the features() function.
        dim is the string specifying the dimensions to extract for the data.

        """
        if dim in self.attributes:
            return self.attributes[dim](cluster_id, load_all=load_all)
        masks = bunch.get('masks', None)
        channel_id, pc = self._get_channel_and_pc(dim)
        # Skip the plot if the channel id is not displayed.
        if channel_id not in bunch.channel_ids:  # pragma: no cover
            return Bunch(data=np.zeros((bunch.data.shape[0],)))
        # Get the column index of the current channel in data.
        c = list(bunch.channel_ids).index(channel_id)
        if masks is not None:
            masks = masks[:, c]
        return Bunch(data=self.feature_scaling * bunch.data[:, c, pc], masks=masks)

    def _get_axis_bounds(self, dim, bunch):
        """Return the min/max of an axis."""
        if dim in self.attributes:
            # Attribute: specified lim, or compute the min/max.
            vmin, vmax = bunch.get('lim', (0, 0))
            assert vmin is not None
            assert vmax is not None
            return vmin, vmax
        return (-self._lim, +self._lim)

    def _plot_points(self, bunch, clu_idx=None):
        if not bunch:
            return
        cluster_id = self.cluster_ids[clu_idx] if clu_idx is not None else None
        for i, j, dim_x, dim_y in self._iter_subplots():
            px = self._get_axis_data(bunch, dim_x, cluster_id=cluster_id)
            py = self._get_axis_data(bunch, dim_y, cluster_id=cluster_id)
            # Skip empty data.
            if px is None or py is None:  # pragma: no cover
                logger.warning("Skipping empty data for cluster %d.", cluster_id)
                return
            assert px.data.shape == py.data.shape
            xmin, xmax = self._get_axis_bounds(dim_x, px)
            ymin, ymax = self._get_axis_bounds(dim_y, py)
            assert xmin <= xmax
            assert ymin <= ymax
            data_bounds = (xmin, ymin, xmax, ymax)
            masks = _get_masks_max(px, py)
            # Prepare the batch visual with all subplots
            # for the selected cluster.
            self.visual.add_batch_data(
                x=px.data, y=py.data,
                color=_get_point_color(clu_idx),
                # Reduced marker size for background features
                size=self._marker_size,
                masks=_get_point_masks(clu_idx=clu_idx, masks=masks),
                data_bounds=data_bounds,
                box_index=(i, j),
            )
            # Get the channel ids corresponding to the relative channel indices
            # specified in the dimensions. Channel 0 corresponds to the first
            # best channel for the selected cluster, and so on.
            label_x = self._get_axis_label(dim_x)
            label_y = self._get_axis_label(dim_y)
            # Add labels.
            self.text_visual.add_batch_data(
                pos=[1, 1],
                anchor=[-1, -1],
                text=label_y,
                data_bounds=None,
                box_index=(i, j),
            )
            self.text_visual.add_batch_data(
                pos=[0, -1.],
                anchor=[0, 1],
                text=label_x,
                data_bounds=None,
                box_index=(i, j),
            )

    def _plot_axes(self):
        self.line_visual.reset_batch()
        for i, j, dim_x, dim_y in self._iter_subplots():
            self.line_visual.add_batch_data(
                pos=[[-1., 0., +1., 0.],
                     [0., -1., 0., +1.]],
                color=(.5, .5, .5, .5),
                box_index=(i, j),
                data_bounds=None,
            )
        self.canvas.update_visual(self.line_visual)

    def _get_lim(self, bunchs):
        if not bunchs:  # pragma: no cover
            return 1
        m, M = min(bunch.data.min() for bunch in bunchs), max(bunch.data.max() for bunch in bunchs)
        M = max(abs(m), abs(M))
        return M

    def _get_scaling_value(self):
        return self.feature_scaling

    def _set_scaling_value(self, value):
        self.feature_scaling = value
        self.plot(fixed_channels=True)

    # Public methods
    # -------------------------------------------------------------------------

    def clear_channels(self):
        """Reset the current channels."""
        self.channel_ids = None
        self.plot()

    def get_clusters_data(self, fixed_channels=None, load_all=None):
        # Get the feature data.
        # Specify the channel ids if these are fixed, otherwise
        # choose the first cluster's best channels.
        c = self.channel_ids if fixed_channels else None
        bunchs = [self.features(cluster_id, channel_ids=c) for cluster_id in self.cluster_ids]
        bunchs = [b for b in bunchs if b]
        if not bunchs:  # pragma: no cover
            return []
        for cluster_id, bunch in zip(self.cluster_ids, bunchs):
            bunch.cluster_id = cluster_id

        # Choose the channels based on the first selected cluster.
        channel_ids = list(bunchs[0].get('channel_ids', [])) if bunchs else []
        common_channels = list(channel_ids)
        # Intersection (with order kept) of channels belonging to all clusters.
        for bunch in bunchs:
            common_channels = [c for c in bunch.get('channel_ids', []) if c in common_channels]
        # The selected channels will be (1) the channels common to all clusters, followed
        # by (2) remaining channels from the first cluster (excluding those already selected
        # in (1)).
        n = len(channel_ids)
        not_common_channels = [c for c in channel_ids if c not in common_channels]
        channel_ids = common_channels + not_common_channels[:n - len(common_channels)]
        assert len(channel_ids) > 0

        # Choose the channels automatically unless fixed_channels is set.
        if (not fixed_channels or self.channel_ids is None):
            self.channel_ids = channel_ids
        assert len(self.channel_ids)

        # Channel labels.
        self.channel_labels = {}
        for d in bunchs:
            chl = d.get('channel_labels', ['%d' % ch for ch in d.get('channel_ids', [])])
            self.channel_labels.update({
                channel_id: chl[i] for i, channel_id in enumerate(d.get('channel_ids', []))})

        return bunchs

    def plot(self, **kwargs):
        """Update the view with the selected clusters."""

        # Determine whether the channels should be fixed or not.
        added = kwargs.get('up', {}).get('added', None)
        # Fix the channels if the view updates after a cluster event
        # and there are new clusters.
        fixed_channels = (
            self.fixed_channels or kwargs.get('fixed_channels', None) or added is not None)

        # Get the clusters data.
        bunchs = self.get_clusters_data(fixed_channels=fixed_channels)
        bunchs = [b for b in bunchs if b]
        if not bunchs:
            return
        self._lim = self._get_lim(bunchs)

        # Get the background data.
        background = self.features(channel_ids=self.channel_ids)

        # Plot all features.
        self._plot_axes()

        # NOTE: the columns in bunch.data are ordered by decreasing quality
        # of the associated channels. The channels corresponding to each
        # column are given in bunch.channel_ids in the same order.

        # Plot points.
        self.visual.reset_batch()
        self.text_visual.reset_batch()

        self._plot_points(background)  # background spikes

        # Plot each cluster.
        for clu_idx, bunch in enumerate(bunchs):
            self._plot_points(bunch, clu_idx=clu_idx)

        # Upload the data on the GPU.
        self.canvas.update_visual(self.visual)
        self.canvas.update_visual(self.text_visual)
        self.canvas.update()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(FeatureView, self).attach(gui)

        self.actions.add(
            self.toggle_automatic_channel_selection,
            checked=not self.fixed_channels, checkable=True)
        self.actions.add(self.clear_channels)
        self.actions.separator()

    def toggle_automatic_channel_selection(self, checked):
        """Toggle the automatic selection of channels when the cluster selection changes."""
        self.fixed_channels = not checked

    @property
    def status(self):
        if self.channel_ids is None:  # pragma: no cover
            return ''
        channel_labels = [self.channel_labels[ch] for ch in self.channel_ids[:2]]
        return 'channels: %s' % ', '.join(channel_labels)

    # Dimension selection
    # -------------------------------------------------------------------------

    def on_select_channel(self, sender=None, channel_id=None, key=None, button=None):
        """Respond to the click on a channel from another view, and update the
        relevant subplots."""
        channels = self.channel_ids
        if channels is None:
            return
        if len(channels) == 1:
            self.plot()
            return
        assert len(channels) >= 2
        # Get the axis from the pressed button (1, 2, etc.)
        if key is not None:
            d = np.clip(len(channels) - 1, 0, key - 1)
        else:
            d = 0 if button == 'Left' else 1
        # Change the first or second best channel.
        old = channels[d]
        # Avoid updating the view if the channel doesn't change.
        if channel_id == old:
            return
        channels[d] = channel_id
        # Ensure that the first two channels are different.
        if channels[1 - min(d, 1)] == channel_id:
            channels[1 - min(d, 1)] = old
        assert channels[0] != channels[1]
        # Remove duplicate channels.
        self.channel_ids = _uniq(channels)
        logger.debug("Choose channels %d and %d in feature view.", *channels[:2])
        # Fix the channels temporarily.
        self.plot(fixed_channels=True)
        self.update_status()

    def on_mouse_click(self, e):
        """Select a feature dimension by clicking on a box in the feature view."""
        b = e.button
        if 'Alt' in e.modifiers:
            # Get mouse position in NDC.
            (i, j), _ = self.canvas.grid.box_map(e.pos)
            dim = self.grid_dim[i][j]
            dim_x, dim_y = dim.split(',')
            dim = dim_x if b == 'Left' else dim_y
            other_dim = dim_y if b == 'Left' else dim_x
            if dim not in self.attributes:
                # When a regular (channel, PC) dimension is selected.
                channel_pc = self._get_channel_and_pc(dim)
                if channel_pc is None:
                    return
                channel_id, pc = channel_pc
                logger.debug("Click on feature dim %s, channel id %s, PC %s.", dim, channel_id, pc)
            else:
                # When the selected dimension is an attribute, e.g. "time".
                pc = None
                # Take the channel id in the other dimension.
                channel_pc = self._get_channel_and_pc(other_dim)
                channel_id = channel_pc[0] if channel_pc is not None else None
                logger.debug("Click on feature dim %s.", dim)
            emit('select_feature', self, dim=dim, channel_id=channel_id, pc=pc)

    def on_request_split(self, sender=None):
        """Return the spikes enclosed by the lasso."""
        if (self.canvas.lasso.count < 3 or not len(self.cluster_ids)):  # pragma: no cover
            return np.array([], dtype=np.int64)
        assert len(self.channel_ids)

        # Get the dimensions of the lassoed subplot.
        i, j = self.canvas.layout.active_box
        dim = self.grid_dim[i][j]
        dim_x, dim_y = dim.split(',')

        # Get all points from all clusters.
        pos = []
        spike_ids = []

        for cluster_id in self.cluster_ids:
            # Load all spikes.
            bunch = self.features(cluster_id, channel_ids=self.channel_ids, load_all=True)
            if not bunch:
                continue
            px = self._get_axis_data(bunch, dim_x, cluster_id=cluster_id, load_all=True)
            py = self._get_axis_data(bunch, dim_y, cluster_id=cluster_id, load_all=True)
            points = np.c_[px.data, py.data]

            # Normalize the points.
            xmin, xmax = self._get_axis_bounds(dim_x, px)
            ymin, ymax = self._get_axis_bounds(dim_y, py)
            r = Range((xmin, ymin, xmax, ymax))
            points = r.apply(points)

            pos.append(points)
            spike_ids.append(bunch.spike_ids)
        pos = np.vstack(pos)
        spike_ids = np.concatenate(spike_ids)

        # Find lassoed spikes.
        ind = self.canvas.lasso.in_polygon(pos)
        self.canvas.lasso.clear()
        return np.unique(spike_ids[ind])
