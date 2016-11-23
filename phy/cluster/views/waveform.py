# -*- coding: utf-8 -*-

"""Waveform view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np
from vispy.util.event import Event

from phy.plot import _get_linear_x
from phy.plot.utils import _get_boxes
from phy.utils import Bunch
from phy.utils._color import _colormap
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------

class ChannelClick(Event):
    def __init__(self, type, channel_idx=None, key=None, button=None):
        super(ChannelClick, self).__init__(type)
        self.channel_idx = channel_idx
        self.key = key
        self.button = button


class WaveformView(ManualClusteringView):
    scaling_coeff = 1.1

    default_shortcuts = {
        'toggle_waveform_overlap': 'o',
        'toggle_zoom_on_channels': 'z',
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

    def __init__(self,
                 waveforms=None,
                 channel_positions=None,
                 channel_order=None,
                 best_channels=None,
                 **kwargs):
        self._key_pressed = None
        self._overlap = False
        self.do_zoom_on_channels = True
        self.do_show_labels = False
        self.filtered_tags = ()

        self.best_channels = best_channels or (lambda clusters: [])

        # Channel positions and n_channels.
        assert channel_positions is not None
        self.channel_positions = np.asarray(channel_positions)
        self.n_channels = self.channel_positions.shape[0]

        # Initialize the view.
        box_bounds = _get_boxes(channel_positions, margin=.1)
        super(WaveformView, self).__init__(layout='boxed',
                                           box_bounds=box_bounds,
                                           **kwargs)

        self.events.add(channel_click=ChannelClick)

        # Box and probe scaling.
        self._box_scaling = np.ones(2)
        self._probe_scaling = np.ones(2)

        # Make a copy of the initial box pos and size. We'll apply the scaling
        # to these quantities.
        self.box_pos = np.array(self.boxed.box_pos)
        self.box_size = np.array(self.boxed.box_size)
        self._update_boxes()

        # Data: functions cluster_id => waveforms.
        self.waveforms = waveforms

        # Channel positions.
        assert channel_positions.shape == (self.n_channels, 2)
        self.channel_positions = channel_positions

        channel_order = (channel_order if channel_order is not None
                         else np.arange(self.n_channels))
        assert channel_order.shape == (self.n_channels,)
        self.channel_order = channel_order

    def on_select(self, cluster_ids=None):
        super(WaveformView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Load the waveform subset.
        data = self.waveforms(cluster_ids)

        # Plot all waveforms.
        with self.building():
            already_shown = set()
            for i, d in enumerate(data):
                if (self.filtered_tags and
                        d.get('tag') not in self.filtered_tags):
                    continue  # pragma: no cover
                alpha = d.get('alpha', .5)
                wave = d.data
                masks = d.masks
                # By default, this is 0, 1, 2 for the first 3 clusters.
                # But it can be customized when displaying several sets
                # of waveforms per cluster.
                pos_idx = cluster_ids.index(d.cluster_id)  # 0, 1, 2, ...

                n_spikes_clu, n_samples, n_unmasked = wave.shape
                assert masks.shape[0] == n_spikes_clu

                # Find the unmasked channels for those spikes.
                unmasked = d.get('channels', np.arange(self.n_channels))
                assert n_unmasked == len(unmasked)
                assert n_unmasked > 0

                # Find the x coordinates.
                t = _get_linear_x(n_spikes_clu * n_unmasked, n_samples)
                if not self.overlap:
                    t = t + 2.5 * (pos_idx - (n_clusters - 1) / 2.)
                    # The total width should not depend on the number of
                    # clusters.
                    t /= n_clusters

                # Get the spike masks.
                m = masks[:, unmasked].reshape((-1, 1))
                # HACK: on the GPU, we get the actual masks with fract(masks)
                # since we add the relative cluster index. We need to ensure
                # that the masks is never 1.0, otherwise it is interpreted as
                # 0.
                m *= .999
                # NOTE: we add the cluster index which is used for the
                # computation of the depth on the GPU.
                m += pos_idx

                color = tuple(_colormap(pos_idx)) + (alpha,)
                assert len(color) == 4

                # Generate the box index (one number per channel).
                box_index = unmasked
                box_index = np.repeat(box_index, n_samples)
                box_index = np.tile(box_index, n_spikes_clu)
                assert box_index.shape == (n_spikes_clu * n_unmasked *
                                           n_samples,)

                # Generate the waveform array.
                wave = np.transpose(wave, (0, 2, 1))
                wave = wave.reshape((n_spikes_clu * n_unmasked, n_samples))

                self.plot(x=t,
                          y=wave,
                          color=color,
                          masks=m,
                          box_index=box_index,
                          data_bounds=None,
                          uniform=True,
                          )
                # Add channel labels.
                if self.do_show_labels:
                    for ch in unmasked:
                        # Skip labels that have already been shown.
                        if ch in already_shown:
                            continue
                        already_shown.add(ch)
                        ch_label = '%d' % self.channel_order[ch]
                        self[ch].text(pos=[t[0, 0], 0.],
                                      text=ch_label,
                                      anchor=[-1.01, -.25],
                                      data_bounds=None,
                                      )

        # Zoom on the best channels when selecting clusters.
        channels = self.best_channels(cluster_ids)
        if channels is not None and self.do_zoom_on_channels:
            self.zoom_on_channels(channels)

    @property
    def state(self):
        return Bunch(box_scaling=tuple(self.box_scaling),
                     probe_scaling=tuple(self.probe_scaling),
                     overlap=self.overlap,
                     do_zoom_on_channels=self.do_zoom_on_channels,
                     do_show_labels=self.do_show_labels,
                     )

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(WaveformView, self).attach(gui)
        self.actions.add(self.toggle_waveform_overlap)
        self.actions.add(self.toggle_zoom_on_channels)
        self.actions.add(self.toggle_show_labels)

        # Box scaling.
        self.actions.add(self.widen)
        self.actions.add(self.narrow)
        self.actions.add(self.increase)
        self.actions.add(self.decrease)

        # Probe scaling.
        self.actions.add(self.extend_horizontally)
        self.actions.add(self.shrink_horizontally)
        self.actions.add(self.extend_vertically)
        self.actions.add(self.shrink_vertically)

        # We forward the event from VisPy to the phy GUI.
        @self.connect
        def on_channel_click(e):
            gui.emit('channel_click',
                     channel_idx=e.channel_idx,
                     key=e.key,
                     button=e.button,
                     )

    # Overlap
    # -------------------------------------------------------------------------

    @property
    def overlap(self):
        return self._overlap

    @overlap.setter
    def overlap(self, value):
        self._overlap = value
        # HACK: temporarily disable automatic zoom on channels when
        # changing the overlap.
        tmp = self.do_zoom_on_channels
        self.do_zoom_on_channels = False
        self.on_select()
        self.do_zoom_on_channels = tmp

    def toggle_waveform_overlap(self):
        """Toggle the overlap of the waveforms."""
        self.overlap = not self.overlap

    # Box scaling
    # -------------------------------------------------------------------------

    def _update_boxes(self):
        self.boxed.update_boxes(self.box_pos * self.probe_scaling,
                                self.box_size * self.box_scaling)

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

    def toggle_zoom_on_channels(self):
        self.do_zoom_on_channels = not self.do_zoom_on_channels

    def toggle_show_labels(self):
        self.do_show_labels = not self.do_show_labels
        tmp = self.do_zoom_on_channels
        self.do_zoom_on_channels = False
        self.on_select()
        self.do_zoom_on_channels = tmp

    def filter_by_tag(self, tag=None):
        """Only show elements with a given tag."""
        self.filtered_tags = (tag,) if tag else ()
        tmp = self.do_zoom_on_channels
        self.do_zoom_on_channels = False
        self.on_select()
        self.do_zoom_on_channels = tmp

    def zoom_on_channels(self, channels_rel):
        """Zoom on some channels."""
        if channels_rel is None or not len(channels_rel):
            return
        channels_rel = np.asarray(channels_rel, dtype=np.int32)
        assert 0 <= channels_rel.min() <= channels_rel.max() < self.n_channels
        # Bounds of the channels.
        b = self.boxed.box_bounds[channels_rel]
        x0, y0 = b[:, :2].min(axis=0)
        x1, y1 = b[:, 2:].max(axis=0)
        # Center of the new range.
        cx = (x0 + x1) * .5
        cy = (y0 + y1) * .5
        # Previous range.
        px0, py0, px1, py1 = self.panzoom.get_range()
        # Half-size of the previous range.
        dx = (px1 - px0) * .5
        dy = (py1 - py0) * .5
        # New range.
        new_range = (cx - dx, cy - dy, cx + dx, cy + dy)
        self.panzoom.set_range(new_range)

    def on_key_press(self, event):
        """Handle key press events."""
        key = event.key
        self._key_pressed = key

    def on_mouse_press(self, e):
        key = self._key_pressed
        if 'Control' in e.modifiers or key in map(str, range(10)):
            key = int(key.name) if key in map(str, range(10)) else None
            # Get mouse position in NDC.
            mouse_pos = self.panzoom.get_mouse_pos(e.pos)
            channel_idx = self.boxed.get_closest_box(mouse_pos)
            self.events.channel_click(channel_idx=channel_idx,
                                      key=key,
                                      button=e.button)

    def on_key_release(self, event):
        self._key_pressed = None
