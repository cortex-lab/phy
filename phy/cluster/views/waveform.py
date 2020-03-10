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
from phy.utils.color import selected_cluster_color
from phy.plot import get_linear_x
from phy.plot.visuals import (  # noqa
    PlotVisual, PlotAggVisual, UniformScatterVisual, TextVisual, LineVisual, _min, _max)
from phy.cluster._utils import RotatingProperty
from .base import ManualClusteringView, ScalingMixin

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------

def _get_box_pos(bunchs, channel_ids):
    cp = {}
    for d in bunchs:
        cp.update({cid: pos for cid, pos in zip(d.channel_ids, d.channel_positions)})
    return np.stack([cp[cid] for cid in channel_ids])


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


class WaveformView(ScalingMixin, ManualClusteringView):
    """This view shows the waveforms of the selected clusters, on relevant channels,
    following the probe geometry.

    Constructor
    -----------

    waveforms : dict of functions
        Every function maps a cluster id to a Bunch with the following attributes:

        * `data` : a 3D array `(n_spikes, n_samples, n_channels_loc)`
        * `channel_ids` : the channel ids corresponding to the third dimension in `data`
        * `channel_labels` : a list of channel labels for every channel in `channel_ids`
        * `channel_positions` : a 2D array with the coordinates of the channels on the probe
        * `masks` : a 2D array `(n_spikes, n_channels)` with the waveforms masks
        * `alpha` : the alpha transparency channel

        The keys of the dictionary are called **waveform types**. The `next_waveforms_type`
        action cycles through all available waveform types. The key `waveforms` is mandatory.
    waveforms_type : str
        Default key of the waveforms dictionary to plot initially.

    """

    # Do not show too many clusters.
    max_n_clusters = 8

    _default_position = 'right'
    ax_color = (.75, .75, .75, 1.)
    tick_size = 5.
    cluster_ids = ()

    default_shortcuts = {
        'toggle_waveform_overlap': 'o',
        'toggle_show_labels': 'ctrl+l',
        'next_waveforms_type': 'w',
        'previous_waveforms_type': 'shift+w',
        'toggle_mean_waveforms': 'm',

        # Box scaling.
        'widen': 'ctrl+right',
        'narrow': 'ctrl+left',
        'increase': 'ctrl+up',
        'decrease': 'ctrl+down',
        'change_box_size': 'ctrl+wheel',

        # Probe scaling.
        'extend_horizontally': 'shift+right',
        'shrink_horizontally': 'shift+left',
        'extend_vertically': 'shift+up',
        'shrink_vertically': 'shift+down',
    }
    default_snippets = {
        'change_n_spikes_waveforms': 'wn',
    }

    def __init__(self, waveforms=None, waveforms_type=None, sample_rate=None, **kwargs):
        self._overlap = False
        self.do_show_labels = True
        self.channel_ids = None
        self.filtered_tags = ()
        self.wave_duration = 0.  # updated in the plotting method
        self.data_bounds = None
        self.sample_rate = sample_rate
        self._status_suffix = ''
        assert sample_rate > 0., "The sample rate must be provided to the waveform view."

        # Initialize the view.
        super(WaveformView, self).__init__(**kwargs)
        self.state_attrs += ('waveforms_type', 'overlap', 'do_show_labels')
        self.local_state_attrs += ('box_scaling', 'probe_scaling')

        # Box and probe scaling.
        self.canvas.set_layout('boxed', box_pos=np.zeros((1, 2)))

        # Ensure waveforms is a dictionary, even if there is a single waveforms type.
        waveforms = waveforms or {}
        waveforms = waveforms if isinstance(waveforms, dict) else {'waveforms': waveforms}
        self.waveforms = waveforms

        # Rotating property waveforms types.
        self.waveforms_types = RotatingProperty()
        for name, value in self.waveforms.items():
            self.waveforms_types.add(name, value)
        # Current waveforms type.
        self.waveforms_types.set(waveforms_type)
        assert self.waveforms_type in self.waveforms

        self.text_visual = TextVisual()
        self.canvas.add_visual(self.text_visual)

        self.line_visual = LineVisual()
        self.canvas.add_visual(self.line_visual)

        self.tick_visual = UniformScatterVisual(
            marker='vbar', color=self.ax_color, size=self.tick_size)
        self.canvas.add_visual(self.tick_visual)

        # Two types of visuals: thin raw line visual for normal waveforms, thick antialiased
        # agg plot visual for mean and template waveforms.
        self.waveform_agg_visual = PlotAggVisual()
        self.waveform_visual = PlotVisual()
        self.canvas.add_visual(self.waveform_agg_visual)
        self.canvas.add_visual(self.waveform_visual)

    # Internal methods
    # -------------------------------------------------------------------------

    @property
    def _current_visual(self):
        if self.waveforms_type == 'waveforms':
            return self.waveform_visual
        else:
            return self.waveform_agg_visual

    def _get_data_bounds(self, bunchs):
        m = min(_min(b.data) for b in bunchs)
        M = max(_max(b.data) for b in bunchs)
        # Symmetrize on the y axis.
        M = max(abs(m), abs(M))
        return [-1, -M, +1, M]

    def get_clusters_data(self):
        if self.waveforms_type not in self.waveforms:
            return
        bunchs = [
            self.waveforms_types.get()(cluster_id) for cluster_id in self.cluster_ids]
        clu_offsets = _get_clu_offsets(bunchs)
        n_clu = max(clu_offsets) + 1
        # Offset depending on the overlap.
        for i, (bunch, offset) in enumerate(zip(bunchs, clu_offsets)):
            bunch.index = i
            bunch.offset = offset
            bunch.n_clu = n_clu
            bunch.color = selected_cluster_color(i, bunch.get('alpha', .75))
        return bunchs

    def _plot_cluster(self, bunch):
        wave = bunch.data
        if wave is None or not wave.size:
            return
        channel_ids_loc = bunch.channel_ids

        n_channels = len(channel_ids_loc)
        masks = bunch.get('masks', np.ones((wave.shape[0], n_channels)))
        # By default, this is 0, 1, 2 for the first 3 clusters.
        # But it can be customized when displaying several sets
        # of waveforms per cluster.

        n_spikes_clu, n_samples = wave.shape[:2]
        assert wave.shape[2] == n_channels
        assert masks.shape == (n_spikes_clu, n_channels)

        # Find the x coordinates.
        t = get_linear_x(n_spikes_clu * n_channels, n_samples)
        t = _overlap_transform(t, offset=bunch.offset, n=bunch.n_clu, overlap=self.overlap)
        # HACK: on the GPU, we get the actual masks with fract(masks)
        # since we add the relative cluster index. We need to ensure
        # that the masks is never 1.0, otherwise it is interpreted as
        # 0.
        eps = .001
        masks = eps + (1 - 2 * eps) * masks
        # NOTE: we add the cluster index which is used for the
        # computation of the depth on the GPU.
        masks += bunch.index

        # Generate the box index (one number per channel).
        box_index = _index_of(channel_ids_loc, self.channel_ids)
        box_index = np.tile(box_index, n_spikes_clu)

        # Find the correct number of vertices depending on the current waveform visual.
        if self._current_visual == self.waveform_visual:
            # PlotVisual
            box_index = np.repeat(box_index, n_samples)
            assert box_index.size == n_spikes_clu * n_channels * n_samples
        else:
            # PlotAggVisual
            box_index = np.repeat(box_index, 2 * (n_samples + 2))
            assert box_index.size == n_spikes_clu * n_channels * 2 * (n_samples + 2)

        # Generate the waveform array.
        wave = np.transpose(wave, (0, 2, 1))
        nw = n_spikes_clu * n_channels
        wave = wave.reshape((nw, n_samples))

        assert self.data_bounds is not None
        self._current_visual.add_batch_data(
            x=t, y=wave, color=bunch.color, masks=masks, box_index=box_index,
            data_bounds=self.data_bounds)

        # Waveform axes.
        # --------------

        # Horizontal y=0 lines.
        ax_db = self.data_bounds
        a, b = _overlap_transform(
            np.array([-1, 1]), offset=bunch.offset, n=bunch.n_clu, overlap=self.overlap)
        box_index = _index_of(channel_ids_loc, self.channel_ids)
        box_index = np.repeat(box_index, 2)
        box_index = np.tile(box_index, n_spikes_clu)
        hpos = np.tile([[a, 0, b, 0]], (nw, 1))
        assert box_index.size == hpos.shape[0] * 2
        self.line_visual.add_batch_data(
            pos=hpos,
            color=self.ax_color,
            data_bounds=ax_db,
            box_index=box_index,
        )

        # Vertical ticks every millisecond.
        steps = np.arange(np.round(self.wave_duration * 1000))
        # A vline every millisecond.
        x = .001 * steps
        # Scale to [-1, 1], same coordinates as the waveform points.
        x = -1 + 2 * x / self.wave_duration
        # Take overlap into account.
        x = _overlap_transform(x, offset=bunch.offset, n=bunch.n_clu, overlap=self.overlap)
        x = np.tile(x, len(channel_ids_loc))
        # Generate the box index.
        box_index = _index_of(channel_ids_loc, self.channel_ids)
        box_index = np.repeat(box_index, x.size // len(box_index))
        assert x.size == box_index.size
        self.tick_visual.add_batch_data(
            x=x, y=np.zeros_like(x),
            data_bounds=ax_db,
            box_index=box_index,
        )

    def _plot_labels(self, channel_ids, n_clusters, channel_labels):
        # Add channel labels.
        if not self.do_show_labels:
            return
        self.text_visual.reset_batch()
        for i, ch in enumerate(channel_ids):
            label = channel_labels[ch]
            self.text_visual.add_batch_data(
                pos=[-1, 0],
                text=str(label),
                anchor=[-1.25, 0],
                box_index=i,
            )
        self.canvas.update_visual(self.text_visual)

    def plot(self, **kwargs):
        """Update the view with the current cluster selection."""
        if not self.cluster_ids:
            return
        bunchs = self.get_clusters_data()
        if not bunchs:
            return

        # All channel ids appearing in all selected clusters.
        channel_ids = sorted(set(_flatten([d.channel_ids for d in bunchs])))
        self.channel_ids = channel_ids
        if bunchs[0].data is not None:
            self.wave_duration = bunchs[0].data.shape[1] / float(self.sample_rate)
        else:  # pragma: no cover
            self.wave_duration = 1.

        # Channel labels.
        channel_labels = {}
        for d in bunchs:
            chl = d.get('channel_labels', ['%d' % ch for ch in d.channel_ids])
            channel_labels.update({
                channel_id: chl[i] for i, channel_id in enumerate(d.channel_ids)})

        # Update the Boxed box positions as a function of the selected channels.
        if channel_ids:
            self.canvas.boxed.update_boxes(_get_box_pos(bunchs, channel_ids))

        self.data_bounds = self.data_bounds or self._get_data_bounds(bunchs)

        self._current_visual.reset_batch()
        self.line_visual.reset_batch()
        self.tick_visual.reset_batch()
        for bunch in bunchs:
            self._plot_cluster(bunch)
        self.canvas.update_visual(self.tick_visual)
        self.canvas.update_visual(self.line_visual)
        self.canvas.update_visual(self._current_visual)

        self._plot_labels(channel_ids, len(self.cluster_ids), channel_labels)

        # Only show the current waveform visual.
        if self._current_visual == self.waveform_visual:
            self.waveform_visual.show()
            self.waveform_agg_visual.hide()
        elif self._current_visual == self.waveform_agg_visual:
            self.waveform_agg_visual.show()
            self.waveform_visual.hide()

        self.canvas.update()
        self.update_status()

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(WaveformView, self).attach(gui)

        self.actions.add(self.toggle_waveform_overlap, checkable=True, checked=self.overlap)
        self.actions.add(self.toggle_show_labels, checkable=True, checked=self.do_show_labels)
        self.actions.add(self.next_waveforms_type)
        self.actions.add(self.previous_waveforms_type)
        self.actions.add(self.toggle_mean_waveforms, checkable=True)
        self.actions.separator()

        # Box scaling.
        self.actions.add(self.widen)
        self.actions.add(self.narrow)
        self.actions.separator()

        # Probe scaling.
        self.actions.add(self.extend_horizontally)
        self.actions.add(self.shrink_horizontally)
        self.actions.separator()

        self.actions.add(self.extend_vertically)
        self.actions.add(self.shrink_vertically)
        self.actions.separator()

    @property
    def boxed(self):
        """Layout instance."""
        return self.canvas.boxed

    @property
    def status(self):
        return self.waveforms_type

    # Overlap
    # -------------------------------------------------------------------------

    @property
    def overlap(self):
        """Whether to overlap the waveforms belonging to different clusters."""
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        self._overlap = value
        self.plot()

    def toggle_waveform_overlap(self, checked):
        """Toggle the overlap of the waveforms."""
        self.overlap = checked

    # Box scaling
    # -------------------------------------------------------------------------

    def widen(self):
        """Increase the horizontal scaling of the waveforms."""
        self.boxed.expand_box_width()

    def narrow(self):
        """Decrease the horizontal scaling of the waveforms."""
        self.boxed.shrink_box_width()

    @property
    def box_scaling(self):
        return self.boxed._box_scaling

    @box_scaling.setter
    def box_scaling(self, value):
        self.boxed._box_scaling = value

    def _get_scaling_value(self):
        return self.boxed._box_scaling[1]

    def _set_scaling_value(self, value):
        w, h = self.boxed._box_scaling
        self.boxed._box_scaling = (w, value)
        self.boxed.update()

    # Probe scaling
    # -------------------------------------------------------------------------

    @property
    def probe_scaling(self):
        return self.boxed._layout_scaling

    @probe_scaling.setter
    def probe_scaling(self, value):
        self.boxed._layout_scaling = value

    def extend_horizontally(self):
        """Increase the horizontal scaling of the probe."""
        self.boxed.expand_layout_width()

    def shrink_horizontally(self):
        """Decrease the horizontal scaling of the waveforms."""
        self.boxed.shrink_layout_width()

    def extend_vertically(self):
        """Increase the vertical scaling of the waveforms."""
        self.boxed.expand_layout_height()

    def shrink_vertically(self):
        """Decrease the vertical scaling of the waveforms."""
        self.boxed.shrink_layout_height()

    # Navigation
    # -------------------------------------------------------------------------

    def toggle_show_labels(self, checked):
        """Whether to show the channel ids or not."""
        self.do_show_labels = checked
        self.text_visual.show() if checked else self.text_visual.hide()
        self.canvas.update()

    def on_mouse_click(self, e):
        """Select a channel by clicking on a box in the waveform view."""
        b = e.button
        nums = tuple('%d' % i for i in range(10))
        if 'Control' in e.modifiers or e.key in nums:
            key = int(e.key) if e.key in nums else None
            # Get mouse position in NDC.
            channel_idx, _ = self.canvas.boxed.box_map(e.pos)
            channel_id = self.channel_ids[channel_idx]
            logger.debug("Click on channel_id %d with key %s and button %s.", channel_id, key, b)
            emit('select_channel', self, channel_id=channel_id, key=key, button=b)

    @property
    def waveforms_type(self):
        return self.waveforms_types.current

    @waveforms_type.setter
    def waveforms_type(self, value):
        self.waveforms_types.set(value)

    def next_waveforms_type(self):
        """Switch to the next waveforms type."""
        self.waveforms_types.next()
        logger.debug("Switch to waveforms type %s.", self.waveforms_type)
        self.plot()

    def previous_waveforms_type(self):
        """Switch to the previous waveforms type."""
        self.waveforms_types.previous()
        logger.debug("Switch to waveforms type %s.", self.waveforms_type)
        self.plot()

    def toggle_mean_waveforms(self, checked):
        """Switch to the `mean_waveforms` type, if it is available."""
        if self.waveforms_type == 'mean_waveforms' and 'waveforms' in self.waveforms:
            self.waveforms_types.set('waveforms')
            logger.debug("Switch to raw waveforms.")
            self.plot()
        elif 'mean_waveforms' in self.waveforms:
            self.waveforms_types.set('mean_waveforms')
            logger.debug("Switch to mean waveforms.")
            self.plot()
