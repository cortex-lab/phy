# -*- coding: utf-8 -*-

"""Waveform view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import defaultdict
import logging

import numpy as np

from phy.plot import _get_linear_x
from phy.plot.interact import Boxed
from phylib.io.array import _flatten, _index_of
from phylib.utils import emit
from phylib.utils._color import _colormap
from phylib.utils.geometry import _get_boxes
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------

def _get_box_bounds(bunchs, channel_ids):
    cp = {}
    for d in bunchs:
        cp.update({cid: pos
                   for cid, pos in zip(d.channel_ids,
                                       d.channel_positions)})
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
        # For very cluster, find the largest existing offset of its channels.
        offset = max(n_clu_per_channel[ch] for ch in bunch.channel_ids)
        offsets.append(offset)
        # Increase the offsets of all channels in the cluster.
        for ch in bunch.channel_ids:
            n_clu_per_channel[ch] += 1

    return offsets


class WaveformView(ManualClusteringView):
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

        self._box_scaling = np.ones(2)
        self._probe_scaling = np.ones(2)

        self.box_pos = np.array(self.canvas.boxed.box_pos)
        self.box_size = np.array(self.canvas.boxed.box_size)
        self._update_boxes()

        # Data: functions cluster_id => waveforms.
        self.waveforms = waveforms

    def _get_data_bounds(self, bunchs):
        m = min(b.data.min() for b in bunchs)
        M = max(b.data.max() for b in bunchs)
        return [-1, m, +1, M]

    def _plot_labels(self, channel_ids, n_clusters, channel_labels=None, data_bounds=None):
        # Add channel labels.
        if self.do_show_labels:
            # Label positions.
            if not self.overlap:
                x = -1 - 2.5 * (n_clusters - 1) / 2.
                x /= n_clusters
            else:
                x = -1.
            for i, ch in enumerate(channel_ids):
                label = (self.channel_labels[i]
                         if self.channel_labels is not None else ch)
                self.canvas[i].text_batch(
                    pos=[x, 0.],
                    text=str(label),
                    anchor=[-1.01, -.25],
                    data_bounds=data_bounds,
                )
            self.canvas.text()

    def _plot_waveforms(self, bunchs, channel_ids, data_bounds=None):
        # Initialize the box scaling the first time.
        if self.box_scaling[1] == 1.:
            M = np.max([np.max(np.abs(b.data)) for b in bunchs])
            self.box_scaling[1] = 1. / M
            self._update_boxes()
        clu_offsets = _get_clu_offsets(bunchs)
        max_clu_offsets = max(clu_offsets) + 1
        for i, d in enumerate(bunchs):
            wave = d.data
            alpha = d.get('alpha', .75)
            channel_ids_loc = d.channel_ids

            n_channels = len(channel_ids_loc)
            masks = d.get('masks', np.ones((wave.shape[0], n_channels)))
            # By default, this is 0, 1, 2 for the first 3 clusters.
            # But it can be customized when displaying several sets
            # of waveforms per cluster.
            # i = cluster_ids.index(d.cluster_id)  # 0, 1, 2, ...

            n_spikes_clu, n_samples = wave.shape[:2]
            assert wave.shape[2] == n_channels
            assert masks.shape == (n_spikes_clu, n_channels)

            # Find the x coordinates.
            t = _get_linear_x(n_spikes_clu * n_channels, n_samples)
            if not self.overlap:

                # Determine the cluster offset.
                offset = clu_offsets[i]
                t = t + 2.5 * (offset - (max_clu_offsets - 1) / 2.)
                # The total width should not depend on the number of
                # clusters.
                t /= max_clu_offsets

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

            color = tuple(_colormap(i)) + (alpha,)
            assert len(color) == 4

            # Generate the box index (one number per channel).
            box_index = _index_of(channel_ids_loc, channel_ids)
            box_index = np.repeat(box_index, n_samples)
            box_index = np.tile(box_index, n_spikes_clu)
            assert box_index.shape == (n_spikes_clu *
                                       n_channels *
                                       n_samples,)

            # Generate the waveform array.
            wave = np.transpose(wave, (0, 2, 1))
            wave = wave.reshape((n_spikes_clu * n_channels, n_samples))
            self._waveform_min_max = (wave.min(), wave.max())

            self.canvas.uplot(
                x=t,
                y=wave,
                color=color,
                masks=m,
                box_index=box_index,
                data_bounds=data_bounds,
            )

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        n_clusters = len(cluster_ids)
        if not cluster_ids:
            return

        # Retrieve the waveform data.
        bunchs = [self.waveforms(cluster_id)
                  for cluster_id in cluster_ids]
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

        self.canvas.clear()
        data_bounds = self._get_data_bounds(bunchs)
        self._plot_waveforms(bunchs, channel_ids, data_bounds=data_bounds)
        self._plot_labels(channel_ids, n_clusters, data_bounds=data_bounds)

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

        # TODO: channel_click event

    # Overlap
    # -------------------------------------------------------------------------

    @property
    def overlap(self):
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
        return self.canvas.boxed

    @property
    def box_scaling(self):
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
        self.do_show_labels = checked
        self.on_select(cluster_ids=self.cluster_ids)

    def on_mouse_click(self, e):
        b = e.button
        nums = tuple('%d' % i for i in range(10))
        if 'Control' in e.modifiers or e.key in nums:
            key = int(e.key) if e.key in nums else None
            # Get mouse position in NDC.
            channel_idx, _ = self.canvas.boxed.box_map(e.pos)
            channel_id = self.channel_ids[channel_idx]
            logger.debug("Click on channel %d with key %s and button %s.", channel_id, key, b)
            emit('channel_click', self, channel_id=channel_id, key=key, button=b)
