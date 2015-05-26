# -*- coding: utf-8 -*-

"""Views for Kwik model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.array import _spikes_in_clusters
from ...stats.ccg import correlograms, _symmetrize_correlograms
from ..ccg import CorrelogramView
from ..features import FeatureView
from ..waveforms import WaveformView
from ..traces import TraceView
from .base import _selected_clusters_colors, BaseViewModel


#------------------------------------------------------------------------------
# Misc functions
#------------------------------------------------------------------------------

def _oddify(x):
    return x if x % 2 == 1 else x + 1


#------------------------------------------------------------------------------
# View models
#------------------------------------------------------------------------------

class WaveformViewModel(BaseViewModel):
    _view_class = WaveformView
    _view_name = 'waveforms'
    _imported_params = ('scale_factor', 'box_scale', 'probe_scale',
                        'overlap', 'show_mean')

    def on_open(self):
        super(WaveformViewModel, self).on_open()
        # Waveforms.
        self.view.visual.channel_positions = self.model.probe.positions
        self.view.visual.channel_order = self.model.channel_order
        # Mean waveforms.
        self.view.mean.channel_positions = self.model.probe.positions
        self.view.mean.channel_order = self.model.channel_order
        if self.scale_factor is None:
            self.scale_factor = 1.

    def _load_waveforms(self):
        # NOTE: we load all spikes from the store.
        # The waveforms store item is responsible for making a subselection
        # of spikes both on disk and in the view.
        waveforms = self.store.load('waveforms',
                                    clusters=self.cluster_ids,
                                    )
        return waveforms

    def _load_mean_waveforms(self):
        mean_waveforms = self.store.load('mean_waveforms',
                                         clusters=self.cluster_ids,
                                         )
        mean_masks = self.store.load('mean_masks',
                                     clusters=self.cluster_ids,
                                     )
        return mean_waveforms, mean_masks

    def _update_spike_clusters(self, spikes=None):
        super(WaveformViewModel, self)._update_spike_clusters(spikes=spikes)
        self._view.mean.spike_clusters = np.sort(self.cluster_ids)
        self._view.mean.cluster_colors = self._view.visual.cluster_colors

    def on_select(self):
        # Get the spikes of the stored waveforms.
        clusters = self.cluster_ids
        n_clusters = len(clusters)
        waveforms = self._load_waveforms()
        spikes = self.store.items['waveforms'].spikes_in_clusters(clusters)
        n_spikes = len(spikes)
        _, self._n_samples, self._n_channels = waveforms.shape
        mean_waveforms, mean_masks = self._load_mean_waveforms()

        self._update_spike_clusters(spikes)

        # Cluster display order.
        self.view.visual.cluster_order = clusters
        self.view.mean.cluster_order = clusters

        # Waveforms.
        assert waveforms.shape[0] == n_spikes
        self.view.visual.waveforms = waveforms * self.scale_factor

        assert mean_waveforms.shape == (n_clusters,
                                        self._n_samples,
                                        self._n_channels)
        self.view.mean.waveforms = mean_waveforms * self.scale_factor

        # Masks.
        masks = self.store.load('masks', clusters=clusters, spikes=spikes)
        assert masks.shape == (n_spikes, self._n_channels)
        self.view.visual.masks = masks

        assert mean_masks.shape == (n_clusters, self._n_channels)
        self.view.mean.masks = mean_masks

        # Spikes.
        self.view.visual.spike_ids = spikes
        self.view.mean.spike_ids = np.arange(len(clusters))

        self.view.update()

    def on_close(self):
        self.view.visual.channel_positions = []
        self.view.mean.channel_positions = []
        super(WaveformViewModel, self).on_close()

    @property
    def box_scale(self):
        """Scale of the waveforms.

        This is a pair of scalars.

        """
        return self.view.box_scale

    @box_scale.setter
    def box_scale(self, value):
        self.view.box_scale = value

    @property
    def probe_scale(self):
        """Scale of the probe.

        This is a pair of scalars.

        """
        return self.view.probe_scale

    @probe_scale.setter
    def probe_scale(self, value):
        self.view.probe_scale = value

    @property
    def overlap(self):
        """Whether to overlap waveforms."""
        return self.view.overlap

    @overlap.setter
    def overlap(self, value):
        self.view.overlap = value

    @property
    def show_mean(self):
        """Whether to show mean waveforms."""
        return self.view.show_mean

    @show_mean.setter
    def show_mean(self, value):
        self.view.show_mean = value

    def exported_params(self, save_size_pos=True):
        params = super(WaveformViewModel, self).exported_params(save_size_pos)
        params.update({
            'scale_factor': self.scale_factor,
            'box_scale': self.view.box_scale,
            'probe_scale': self.view.probe_scale,
            'overlap': self.view.overlap,
            'show_mean': self.view.show_mean,
        })
        return params


class FeatureViewModel(BaseViewModel):
    _view_class = FeatureView
    _view_name = 'features'
    _imported_params = ('scale_factor', 'n_spikes_max_bg', 'marker_size')
    n_spikes_max_bg = 10000

    def __init__(self, **kwargs):
        self._dimension_selector = None
        self._previous_dimensions = None
        self._previous_pan_zoom = None
        super(FeatureViewModel, self).__init__(**kwargs)
        self._view.connect(self.on_mouse_double_click)

    def set_dimension_selector(self, func):
        """Decorator for a function that selects the best projection.

        The decorated function must have the following signature:

        ```python
        @view_model.set_dimension_selector
        def choose(cluster_ids):
            # ...
            return channel_idxs  # a list with 3 relative channel indices
        ```

        """
        self._dimension_selector = func

    def default_dimension_selector(self, cluster_ids):
        """Return the channels with the largest mean features.

        The first cluster is used currently.

        """
        if cluster_ids is None or not len(cluster_ids):
            return np.arange(len(self.model.channels[:3]))
        n_fet = self.model.n_features_per_channel
        score = self.store.mean_features(cluster_ids[0])
        score = score.reshape((-1, n_fet)).mean(axis=1)
        # Take the best 3 channels.
        assert len(score) == len(self.model.channel_order)
        channels = np.argsort(score)[::-1][:3]
        return channels

    def _default_dimensions(self, cluster_ids=None):
        dimension_selector = (self._dimension_selector or
                              self.default_dimension_selector)
        channels = dimension_selector(cluster_ids)
        return ['time'] + [(ch, 0) for ch in channels]

    def _rescale_features(self, features):
        # WARNING: convert features to a 3D array
        # (n_spikes, n_channels, n_features)
        # because that's what the FeatureView expects currently.
        n_fet = self.model.n_features_per_channel
        n_channels = len(self.model.channel_order)
        shape = (-1, n_channels, n_fet)
        features = features[:, :n_fet * n_channels].reshape(shape)
        # Scale factor.
        return features * self.scale_factor

    @property
    def lasso(self):
        return self.view.lasso

    def spikes_in_lasso(self):
        """Return the spike ids from the selected clusters within the lasso."""
        if not len(self.cluster_ids) or self.view.lasso.n_points <= 2:
            return
        clusters = self.cluster_ids
        features = self.store.load('features', clusters=clusters)
        features = self._rescale_features(features)
        box = self.view.lasso.box
        points = self.view.visual.project(features, box)
        in_lasso = self.view.lasso.in_lasso(points)
        spike_ids = _spikes_in_clusters(self.model.spike_clusters, clusters)
        return spike_ids[in_lasso]

    @property
    def marker_size(self):
        """Marker size."""
        return self.view.marker_size

    @marker_size.setter
    def marker_size(self, value):
        self.view.marker_size = value

    @property
    def dimensions(self):
        """The list of displayed dimensions."""
        return self._view.dimensions

    @dimensions.setter
    def dimensions(self, value):
        self._view.dimensions = value

    @property
    def diagonal_dimensions(self):
        """The list of dimensions on the diagonal (y axis)."""
        return self._view.diagonal_dimensions

    @diagonal_dimensions.setter
    def diagonal_dimensions(self, value):
        self._view.diagonal_dimensions = value

    def on_open(self):
        # Get background features.
        # TODO OPTIM: precompute this once for all and store in the cluster
        # store. But might be unnecessary.
        if self.n_spikes_max_bg is not None:
            k = max(1, self.model.n_spikes // self.n_spikes_max_bg)
        else:
            k = 1
        if self.model.features is not None:
            # Background features.
            features_bg = self.store.load('features',
                                          spikes=slice(None, None, k))
            self.view.background.features = self._rescale_features(features_bg)
            # Time dimension.
            spike_samples = self.model.spike_samples[::k]
            self.view.background.spike_samples = spike_samples
        self.view.update_dimensions(self._default_dimensions())

    def on_select(self):
        super(FeatureViewModel, self).on_select()
        spikes = self.spike_ids
        clusters = self.cluster_ids

        features = self.store.load('features',
                                   clusters=clusters,
                                   spikes=spikes)
        masks = self.store.load('masks',
                                clusters=clusters,
                                spikes=spikes)

        nc = len(self.model.channel_order)
        nf = self.model.n_features_per_channel
        features = features.reshape((len(spikes), nc, nf))
        self.view.visual.features = self._rescale_features(features)
        self.view.visual.masks = masks

        # Spikes.
        self.view.visual.spike_ids = spikes
        self.view.visual.spike_samples = self.model.spike_samples[spikes]

        # Cluster display order.
        self.view.visual.cluster_order = clusters

        # Choose best projection.
        self.view.dimensions = self._default_dimensions(clusters)

        self.view.update()

    keyboard_shortcuts = {
        'enlarge_subplot': 'double left click',
    }

    def on_mouse_double_click(self, e):
        if self._previous_dimensions:
            self.dimensions = self._previous_dimensions
            self._previous_dimensions = None

            p, z = self._previous_pan_zoom
            self._view._pz._index = (0, 0)
            self._view._pz.pan = p
            self._view._pz.zoom = z
            self._previous_pan_zoom = None

            self._view._pz._index = self._previous_index
            self._previous_index = None
        else:
            # Find the current box.
            i, j = self._view._pz._get_box(e.pos)
            # Save previous (diagonal) dimensions.
            self._previous_dimensions = self.dimensions
            self._previous_index = (i, j)
            # Save the previous pan-zoom of the first subplot (which is
            # going to be replaced).
            self._previous_pan_zoom = (self._view._pz.pan_matrix[0, 0],
                                       self._view._pz.zoom_matrix[0, 0])
            p = self._view._pz.pan
            z = self._view._pz.zoom
            dim_i = self.dimensions[i]
            dim_j = self.dimensions[j]
            # Set the dimensions.
            self.dimensions = [dim_i]
            if i != j:
                self.diagonal_dimensions = [dim_j]
            self._view._pz._index = (0, 0)
            self._view._pz.pan = p
            self._view._pz.zoom = z
            # Update the pan zoom of the new subplot.
        self._view.update()

    def exported_params(self, save_size_pos=True):
        params = super(FeatureViewModel, self).exported_params(save_size_pos)
        zoom = self._view._pz.zoom
        params.update({
            'scale_factor': zoom.mean() * self.scale_factor,
            'marker_size': self.marker_size,
        })
        return params


class CorrelogramViewModel(BaseViewModel):
    _view_class = CorrelogramView
    _view_name = 'correlograms'
    binsize = 20
    winsize_bins = 41
    _imported_params = ('binsize', 'winsize_bins')

    def change_bins(self, bin=None, half_width=None):
        """Change the parameters of the correlograms.

        Parameters
        ----------
        bin : float (ms)
            Bin size.
        half_width : float (ms)
            Half window size.

        """
        sr = self.model.sample_rate

        bin = np.clip(bin * .001, .001, 1e6)
        self.binsize = int(sr * bin)

        half_width = np.clip(half_width * .001, .001, 1e6)
        self.winsize_bins = 2 * int(half_width / bin) + 1

        self.select(self.cluster_ids)

    def on_select(self):
        super(CorrelogramViewModel, self).on_select()
        spikes = self.spike_ids
        clusters = self.cluster_ids

        self.view.cluster_ids = clusters

        # Compute the correlograms.
        spike_samples = self.model.spike_samples[spikes]
        spike_clusters = self.view.visual.spike_clusters

        ccgs = correlograms(spike_samples,
                            spike_clusters,
                            cluster_order=clusters,
                            binsize=self.binsize,
                            # NOTE: this must be an odd number, for symmetry
                            winsize_bins=_oddify(self.winsize_bins),
                            )
        ccgs = _symmetrize_correlograms(ccgs)
        # Normalize the CCGs.
        if len(ccgs):
            ccgs = ccgs * (1. / max(1., ccgs.max()))
        self.view.visual.correlograms = ccgs

        # Take the cluster order into account.
        self.view.visual.cluster_order = clusters
        self.view.update()


class TraceViewModel(BaseViewModel):
    _view_class = TraceView
    _view_name = 'traces'
    _imported_params = ('scale_factor', 'channel_scale', 'interval_size')
    interval_size = .25

    def __init__(self, **kwargs):
        self._interval = None
        super(TraceViewModel, self).__init__(**kwargs)

    def _load_traces(self, interval):
        start, end = interval
        spikes = self.spike_ids

        # Load the traces.
        # debug("Loading traces...")
        # Using channel_order ensures that we get rid of the dead channels.
        # We also keep the channel order as specified by the PRM file.
        # WARNING: HDF5 does not support out-of-order indexing (...!!)
        traces = self.model.traces[start:end, :][:, self.model.channel_order]

        # Normalize and set the traces.
        traces_f = np.empty_like(traces, dtype=np.float32)
        traces_f[...] = traces * self.scale_factor
        # Detrend the traces.
        m = np.mean(traces_f[::10, :], axis=0)
        traces_f -= m
        self.view.visual.traces = traces_f

        # Keep the spikes in the interval.
        spike_samples = self.model.spike_samples[spikes]
        a, b = spike_samples.searchsorted(interval)
        spikes = spikes[a:b]
        self.view.visual.n_spikes = len(spikes)
        self.view.visual.spike_ids = spikes

        if len(spikes) == 0:
            return

        # We update the spike clusters according to the subselection of spikes.
        # We don't update the list of unique clusters, which only change
        # when selecting or clustering, not when changing the interval.
        # self._update_spike_clusters(spikes)
        self.view.visual.spike_clusters = self.model.spike_clusters[spikes]

        # Set the spike samples.
        spike_samples = self.model.spike_samples[spikes]
        # This is in unit of samples relative to the start of the interval.
        spike_samples = spike_samples - start
        self.view.visual.spike_samples = spike_samples
        self.view.visual.offset = start

        # Load the masks.
        # TODO: ensure model.masks is always 2D, even with 1 spike
        masks = np.atleast_2d(self._model.masks[spikes])
        self.view.visual.masks = masks

    @property
    def interval(self):
        """The interval of the view, in unit of sample."""
        return self._interval

    @interval.setter
    def interval(self, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("The interval should be a (start, end) tuple.")
        # Restrict the interval to the boundaries of the traces.
        start, end = value
        start, end = int(start), int(end)
        n = self.model.traces.shape[0]
        if start < 0:
            end += (-start)
            start = 0
        elif end >= n:
            start -= (end - n)
            end = n
        self._interval = (start, end)
        self._load_traces((start, end))
        self.view.update()

    @property
    def channel_scale(self):
        """Vertical scale of the traces."""
        return self.view.channel_scale

    @channel_scale.setter
    def channel_scale(self, value):
        self.view.channel_scale = value

    def move(self, amount):
        """Move the current interval by a given amount (in samples)."""
        amount = int(amount)
        start, end = self.interval
        self.interval = start + amount, end + amount

    def move_right(self, fraction=.05):
        """Move the current interval to the right."""
        start, end = self.interval
        self.move(int(+(end - start) * fraction))

    def move_left(self, fraction=.05):
        """Move the current interval to the left."""
        start, end = self.interval
        self.move(int(-(end - start) * fraction))

    keyboard_shortcuts = {
        'scroll_left': 'ctrl+left',
        'scroll_right': 'ctrl+right',
        'fast_scroll_left': 'shift+left',
        'fast_scroll_right': 'shift+right',
    }

    def on_key_press(self, event):
        super(TraceViewModel, self).on_key_press(event)
        key = event.key
        if 'Control' in event.modifiers:
            if key == 'Left':
                self.move_left()
            elif key == 'Right':
                self.move_right()
        if 'Shift' in event.modifiers:
            if key == 'Left':
                self.move_left(1)
            elif key == 'Right':
                self.move_right(1)

    def on_open(self):
        super(TraceViewModel, self).on_open()
        self.view.visual.n_samples_per_spike = self.model.n_samples_waveforms
        self.view.visual.sample_rate = self.model.sample_rate
        if self.scale_factor is None:
            self.scale_factor = 1.
        if self.interval_size is None:
            self.interval_size = .25
        self.select([])

    def on_select(self):
        # Get the spikes in the selected clusters.
        spikes = self.spike_ids
        clusters = self.cluster_ids
        n_clusters = len(clusters)
        spike_clusters = self.model.spike_clusters[spikes]

        # Update the clusters of the trace view.
        visual = self._view.visual
        visual.spike_clusters = spike_clusters
        visual.cluster_ids = clusters
        visual.cluster_order = clusters
        visual.cluster_colors = _selected_clusters_colors(n_clusters)

        # Select the default interval.
        half_size = int(self.interval_size * self.model.sample_rate / 2.)
        if len(spikes) > 0:
            # Center the default interval around the first spike.
            sample = self._model.spike_samples[spikes[0]]
        else:
            sample = half_size
        # Load traces by setting the interval.
        visual._update_clusters_automatically = False
        self.interval = sample - half_size, sample + half_size

    def exported_params(self, save_size_pos=True):
        params = super(TraceViewModel, self).exported_params(save_size_pos)
        params.update({
            'scale_factor': self.scale_factor,
            'channel_scale': self.channel_scale,
        })
        return params
