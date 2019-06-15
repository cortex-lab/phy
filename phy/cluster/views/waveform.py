# -*- coding: utf-8 -*-

"""Waveform view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
import logging

import numpy as np

from phylib.io.array import _flatten, _index_of
from phylib.utils import emit
from phylib.utils.color import selected_cluster_color
from phylib.utils.geometry import _get_boxes, range_transform
from phy.plot import get_linear_x
from phy.plot.interact import Boxed
from phy.plot.transform import NDC
from phy.plot.visuals import PlotVisual, TextVisual, _min, _max
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------

def _get_box_bounds(bunchs, channel_ids):
    cp = {}
    for d in bunchs:
        cp.update({cid: pos for cid, pos in zip(d.channel_ids, d.channel_positions)})
    box_pos = np.stack([cp[cid] for cid in channel_ids])
    bounds = _get_boxes(box_pos, margin=Boxed.margin)
    return bounds


def _get_clu_offsets(bunchs):
    # Count the number of clusters per channel to determine the x offset
    # of clusters when overlap=False.
    n_clu_per_channel = defaultdict(int)
    offsets = []

    # Determine the offset.
    for bunch in bunchs:
        # For every cluster, find the largest existing offset of its channels.
        offset = max(n_clu_per_channel[ch] for ch in bunch.channel_ids)
        offsets.append(offset)
        # Increase the offsets of all channels in the cluster.
        for ch in bunch.channel_ids:
            n_clu_per_channel[ch] += 1

    return offsets


def _overlap_transform(t, offset=0, n=1, overlap=None):
    if overlap:
        return t
    k = 8  # the waveform size is k times the margin between the margins
    t = -1 + (2 * offset * (k + 1) + (t + 1) * k) / (n * (k + 1) - 1)
    return t


class WaveformView(ManualClusteringView):
    """This view shows the waveforms of the selected clusters, on relevant channels,
    following the probe geometry.

    Constructor
    -----------

    waveforms : function
        Maps a cluster id to a Bunch with the following attributes:
        * `data` : a 3D array `(n_spikes, n_samples, n_channels_loc)`
        * `channel_ids` : the channel ids corresponding to the third dimension in `data
        * `channel_positions` : a 2D array with the coordinates of the channels on the probe
        * `masks` : a 2D array `(n_spikes, n_channels)` with the waveforms masks
        * `alpha` : the alpha transparency channel

    channel_labels : array-like
        Labels of the channels.

    """

    _default_position = 'right'
    scaling_coeff = 1.1
    cluster_ids = ()

    default_shortcuts = {
        'toggle_waveform_overlap': 'o',
        'toggle_show_labels': 'ctrl+l',

        # Box scaling.
        'widen': 'ctrl+right',
        'narrow': 'ctrl+left',
        'increase': 'ctrl+up',
        'decrease': 'ctrl+down',

        # Probe scaling.
        'extend_horizontally': 'shift+right',
        'shrink_horizontally': 'shift+left',
        'extend_vertically': 'shift+up',
        'shrink_vertically': 'shift+down',
    }

    def __init__(self, waveforms=None, channel_labels=None):
        self._overlap = False
        self.do_show_labels = True
        self.channel_ids = None
        self.channel_labels = channel_labels
        self.filtered_tags = ()

        # Initialize the view.
        super(WaveformView, self).__init__()
        self.state_attrs += ('box_scaling', 'probe_scaling', 'overlap', 'do_show_labels')
        self.local_state_attrs += ('box_scaling', 'probe_scaling')

        # Box and probe scaling.
        self.canvas.set_layout('boxed', box_bounds=[[-1, -1, +1, +1]])
        self.canvas.enable_axes()

        self._box_scaling = np.ones(2)
        self._probe_scaling = np.ones(2)

        self.box_pos = np.array(self.canvas.boxed.box_pos)
        self.box_size = np.array(self.canvas.boxed.box_size)
        self._update_boxes()

        # Data: functions cluster_id => waveforms.
        self.waveforms = waveforms

        self.label_visual = TextVisual()
        self.canvas.add_visual(self.label_visual)

        self.waveform_visual = PlotVisual()
        self.canvas.add_visual(self.waveform_visual)

    def _get_data_bounds(self, bunchs):
        m = min(_min(b.data) for b in bunchs)
        M = max(_max(b.data) for b in bunchs)
        return [-1, m, +1, M]

    def _plot_labels(self, channel_ids, n_clusters, channel_labels=None):
        # Add channel labels.
        if not self.do_show_labels:
            return
        self.label_visual.reset_batch()
        for i, ch in enumerate(channel_ids):
            label = self.channel_labels[i] if self.channel_labels is not None else ch
            self.label_visual.add_batch_data(
                pos=[-1, 0],
                text=str(label),
                anchor=[-2, 0],
                box_index=i,
            )
        self.canvas.update_visual(self.label_visual)

    def _plot_waveforms(self, bunchs, channel_ids, data_bounds=None):
        clu_offsets = _get_clu_offsets(bunchs)
        n_clu = max(clu_offsets) + 1
        self.waveform_visual.reset_batch()
        for i, d in enumerate(bunchs):
            wave = d.data
            if not wave.size:
                continue

            alpha = d.get('alpha', .75)
            channel_ids_loc = d.channel_ids

            n_channels = len(channel_ids_loc)
            masks = d.get('masks', np.ones((wave.shape[0], n_channels)))
            # By default, this is 0, 1, 2 for the first 3 clusters.
            # But it can be customized when displaying several sets
            # of waveforms per cluster.

            n_spikes_clu, n_samples = wave.shape[:2]
            assert wave.shape[2] == n_channels
            assert masks.shape == (n_spikes_clu, n_channels)

            # Find the x coordinates.
            t = get_linear_x(n_spikes_clu * n_channels, n_samples)
            t = _overlap_transform(t, offset=clu_offsets[i], n=n_clu, overlap=self.overlap)
            # Get the spike masks.
            m = masks
            # HACK: on the GPU, we get the actual masks with fract(masks)
            # since we add the relative cluster index. We need to ensure
            # that the masks is never 1.0, otherwise it is interpreted as
            # 0.
            m *= .99999
            # NOTE: we add the cluster index which is used for the
            # computation of the depth on the GPU.
            m += i

            color = selected_cluster_color(i, alpha)
            assert len(color) == 4

            # Generate the box index (one number per channel).
            box_index = _index_of(channel_ids_loc, channel_ids)
            box_index = np.repeat(box_index, n_samples)
            box_index = np.tile(box_index, n_spikes_clu)
            assert box_index.shape == (n_spikes_clu * n_channels * n_samples,)

            # Generate the waveform array.
            wave = np.transpose(wave, (0, 2, 1))
            wave = wave.reshape((n_spikes_clu * n_channels, n_samples))

            self.waveform_visual.add_batch_data(
                x=t,
                y=wave,
                color=color,
                masks=m,
                box_index=box_index,
                data_bounds=data_bounds,
            )
        self.canvas.update_visual(self.waveform_visual)

    def on_select(self, cluster_ids=(), **kwargs):
        """Update the view with the selected clusters."""
        self.cluster_ids = cluster_ids
        n_clusters = len(cluster_ids)
        if not cluster_ids:
            return

        # Retrieve the waveform data.
        bunchs = [self.waveforms(cluster_id) for cluster_id in cluster_ids]
        if bunchs[0].data is None:
            return

        # All channel ids appearing in all selected clusters.
        channel_ids = sorted(set(_flatten([d.channel_ids for d in bunchs])))
        box_bounds = _get_box_bounds(bunchs, channel_ids)
        self.channel_ids = channel_ids

        # Update the box bounds as a function of the selected channels.
        self.canvas.boxed.box_bounds = box_bounds
        self.box_pos = np.array(self.canvas.boxed.box_pos)
        self.box_size = np.array(self.canvas.boxed.box_size)
        self._update_boxes()

        data_bounds = self._get_data_bounds(bunchs)
        self._plot_waveforms(bunchs, channel_ids, data_bounds=data_bounds)
        self._plot_labels(channel_ids, n_clusters)

        # Update the axes data bounds.
        _, m, _, M = data_bounds
        # Waveform duration, scaled by overlap factor if needed.
        wave_dur = bunchs[0].get('waveform_duration', 1.)
        wave_dur /= .5 * (1 + _overlap_transform(1, n=n_clusters, overlap=self.overlap))
        x1, y1 = range_transform(box_bounds[0], NDC, [wave_dur, M - m])
        axes_data_bounds = (0, 0, x1, y1)
        self.canvas.axes.reset_data_bounds(axes_data_bounds, do_update=True)
        # self.canvas.boxed.add_boxes(self.canvas)  # for debugging

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(WaveformView, self).attach(gui)
        self.actions.add(self.toggle_waveform_overlap, checkable=True, checked=self.overlap)
        self.actions.add(self.toggle_show_labels, checkable=True, checked=self.do_show_labels)
        self.actions.separator()

        # Box scaling.
        self.actions.add(self.widen)
        self.actions.add(self.narrow)
        self.actions.separator()
        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.separator()

        # Probe scaling.
        self.actions.add(self.extend_horizontally)
        self.actions.add(self.shrink_horizontally)
        self.actions.separator()
        self.actions.add(self.extend_vertically)
        self.actions.add(self.shrink_vertically)

    # Overlap
    # -------------------------------------------------------------------------

    @property
    def overlap(self):
        """Whether to overlap the waveforms belonging to different clusters."""
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        self._overlap = value
        self.on_select(cluster_ids=self.cluster_ids)

    def toggle_waveform_overlap(self, checked):
        """Toggle the overlap of the waveforms."""
        self.overlap = checked

    # Box scaling
    # -------------------------------------------------------------------------

    def _update_boxes(self):
        self.canvas.boxed.update_boxes(
            self.box_pos * self.probe_scaling, self.box_size * self.box_scaling)

    @property
    def boxed(self):
        """Layout instance."""
        return self.canvas.boxed

    @property
    def box_scaling(self):
        """Scaling of the channel boxes."""
        return self._box_scaling

    @box_scaling.setter
    def box_scaling(self, value):
        assert len(value) == 2
        self._box_scaling = np.array(value)
        self._update_boxes()

    def widen(self):
        """Increase the horizontal scaling of the waveforms."""
        self._box_scaling[0] *= self.scaling_coeff
        self._update_boxes()

    def narrow(self):
        """Decrease the horizontal scaling of the waveforms."""
        self._box_scaling[0] /= self.scaling_coeff
        self._update_boxes()

    def increase(self):
        """Increase the vertical scaling of the waveforms."""
        self._box_scaling[1] *= self.scaling_coeff
        self._update_boxes()

    def decrease(self):
        """Decrease the vertical scaling of the waveforms."""
        self._box_scaling[1] /= self.scaling_coeff
        self._update_boxes()

    # Probe scaling
    # -------------------------------------------------------------------------

    @property
    def probe_scaling(self):
        """Scaling of the entire probe."""
        return self._probe_scaling

    @probe_scaling.setter
    def probe_scaling(self, value):
        assert len(value) == 2
        self._probe_scaling = np.array(value)
        self._update_boxes()

    def extend_horizontally(self):
        """Increase the horizontal scaling of the probe."""
        self._probe_scaling[0] *= self.scaling_coeff
        self._update_boxes()

    def shrink_horizontally(self):
        """Decrease the horizontal scaling of the waveforms."""
        self._probe_scaling[0] /= self.scaling_coeff
        self._update_boxes()

    def extend_vertically(self):
        """Increase the vertical scaling of the waveforms."""
        self._probe_scaling[1] *= self.scaling_coeff
        self._update_boxes()

    def shrink_vertically(self):
        """Decrease the vertical scaling of the waveforms."""
        self._probe_scaling[1] /= self.scaling_coeff
        self._update_boxes()

    # Navigation
    # -------------------------------------------------------------------------

    def toggle_show_labels(self, checked):
        """Whether to show the channel ids or not."""
        self.do_show_labels = checked
        self.on_select(cluster_ids=self.cluster_ids)

    def on_mouse_click(self, e):
        """Select a channel by clicking on a box in the waveform view."""
        b = e.button
        nums = tuple('%d' % i for i in range(10))
        if 'Control' in e.modifiers or e.key in nums:
            key = int(e.key) if e.key in nums else None
            # Get mouse position in NDC.
            channel_idx, _ = self.canvas.boxed.box_map(e.pos)
            channel_id = self.channel_ids[channel_idx]
            logger.debug("Click on channel %d with key %s and button %s.", channel_id, key, b)
            emit('channel_click', self, channel_id=channel_id, key=key, button=b)
