# -*- coding: utf-8 -*-

"""Views for Kwik model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.logging import debug
from ...utils.array import _unique
from ...plot.ccg import CorrelogramView
from ...plot.features import FeatureView
from ...plot.waveforms import WaveformView
from ...plot.traces import TraceView
from ...stats.ccg import correlograms, _symmetrize_correlograms
from .selector import Selector
from ._utils import (_update_cluster_selection,
                     _subset_spikes_per_cluster,
                     _concatenate_per_cluster_arrays,
                     )


#------------------------------------------------------------------------------
# BaseViewModel for plot views and Kwik model
#------------------------------------------------------------------------------

def _create_view(cls, backend=None, **kwargs):
    if backend in ('pyqt4', None):
        kwargs.update({'always_on_top': True})
    return cls(**kwargs)


_COLORMAP = np.array([[102, 194, 165],
                      [252, 141, 98],
                      [141, 160, 203],
                      [231, 138, 195],
                      [166, 216, 84],
                      [255, 217, 47],
                      [229, 196, 148],
                      ])


def _selected_clusters_colors(n_clusters):
    if n_clusters > _COLORMAP.shape[0]:
        colors = np.tile(_COLORMAP, (1 + n_clusters // _COLORMAP.shape[0], 1))
    else:
        colors = _COLORMAP
    return colors[:n_clusters, ...] / 255.


class BaseViewModel(object):
    """Used to create views from a model."""
    _view_class = None
    _view_name = ''

    def __init__(self, model, store=None,
                 n_spikes_max=None, excerpt_size=None,
                 **kwargs):
        self._model = model
        self._store = store

        # Create the spike/cluster selector.
        self._selector = Selector(model.spike_clusters,
                                  n_spikes_max=n_spikes_max,
                                  excerpt_size=excerpt_size,
                                  )

        # Set all keyword arguments as attributes.
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Extract VisPy keyword arguments.
        vispy_kwargs_names = ('position', 'size',)
        vispy_kwargs = {name: kwargs[name] for name in vispy_kwargs_names
                        if name in kwargs}
        backend = kwargs.pop('backend', None)

        # Create the VisPy canvas.
        self._view = _create_view(self._view_class,
                                  backend=backend,
                                  **vispy_kwargs)

        # Bind VisPy event methods.
        for method in ('on_key_press', 'on_mouse_move'):
            if hasattr(self, method):
                self._view.connect(getattr(self, method))

    @property
    def model(self):
        return self._model

    @property
    def view_name(self):
        return self._view_name

    @property
    def store(self):
        return self._store

    @property
    def selector(self):
        return self._selector

    @property
    def view(self):
        return self._view

    @property
    def cluster_ids(self):
        return self._selector.selected_clusters

    @property
    def spike_ids(self):
        return self._selector.selected_spikes

    @property
    def n_clusters(self):
        return self._selector.n_clusters

    @property
    def n_spikes(self):
        return self._selector.n_spikes

    def _load_from_store_or_model(self, name, cluster_ids, spikes):
        if self._store is not None:
            return self._store.load(name, cluster_ids, spikes)
        else:
            return getattr(self._model, name)[spikes]

    def _update_spike_clusters(self, spikes=None):
        """Update the spike clusters and cluster colors."""
        if spikes is None:
            spikes = self.spike_ids
            spike_clusters = self.model.spike_clusters[spikes]
            n_clusters = self.n_clusters
        else:
            spike_clusters = self.model.spike_clusters[spikes]
            n_clusters = len(_unique(spike_clusters))
        visual = self._view.visual
        # This updates the list of unique clusters in the view.
        visual.spike_clusters = spike_clusters
        visual.cluster_colors = _selected_clusters_colors(n_clusters)

    def _update_cluster_order(self, up):
        """Update cluster order when a clustering action occurs."""
        self._view.visual.cluster_order = _update_cluster_selection(
            self._view.visual.cluster_order, up)

    def on_open(self):
        """May be overriden."""

    def on_cluster(self, up=None):
        """May be overriden."""
        self._update_spike_clusters()
        self._update_cluster_order(up)

    def on_select(self, cluster_ids):
        """Must be overriden."""
        self._selector.selected_clusters = cluster_ids
        self._update_spike_clusters()

    def on_close(self):
        self._view.visual.spike_clusters = []
        self._view.update()

    def show(self):
        self._view.show()


#------------------------------------------------------------------------------
# View models
#------------------------------------------------------------------------------

class WaveformViewModel(BaseViewModel):
    _view_class = WaveformView
    _view_name = 'waveforms'
    scale_factor = 1.

    def on_open(self):
        super(WaveformViewModel, self).on_open()
        self.view.visual.channel_positions = self.model.probe.positions

    def on_select(self, cluster_ids):
        super(WaveformViewModel, self).on_select(cluster_ids)
        spikes = self.spike_ids

        # Load waveforms.
        debug("Loading {0:d} waveforms...".format(len(spikes)))
        waveforms = self.model.waveforms[spikes]
        debug("Done!")

        # Spikes.
        self.view.visual.spike_ids = spikes

        # Cluster display order.
        self.view.visual.cluster_order = cluster_ids

        # Waveforms.
        waveforms *= self.scale_factor
        self.view.visual.waveforms = waveforms

        # Masks.
        masks = self._load_from_store_or_model('masks', cluster_ids, spikes)
        self.view.visual.masks = masks

    def on_close(self):
        self.view.visual.channel_positions = []
        super(WaveformViewModel, self).on_close()


class FeatureViewModel(BaseViewModel):
    _view_class = FeatureView
    _view_name = 'features'
    scale_factor = 1.

    def on_select(self, cluster_ids):
        super(FeatureViewModel, self).on_select(cluster_ids)
        spikes = self.spike_ids

        # Spikes.
        self.view.visual.spike_ids = spikes
        self.view.visual.spike_samples = self.model.spike_samples[spikes]

        # Cluster display order.
        self.view.visual.cluster_order = cluster_ids

        # Load features.
        features = self._load_from_store_or_model('features',
                                                  cluster_ids,
                                                  spikes)
        # Load masks.
        masks = self._load_from_store_or_model('masks',
                                               cluster_ids,
                                               spikes)

        # WARNING: convert features to a 3D array
        # (n_spikes, n_channels, n_features)
        # because that's what the FeatureView expects currently.
        n_fet = self.model.n_features_per_channel
        n_channels = len(self.model.channel_order)
        shape = (-1, n_channels, n_fet)
        features = features[:, :n_fet * n_channels].reshape(shape)
        # Scale factor.
        features *= self.scale_factor

        self.view.visual.features = features
        self.view.visual.masks = masks

        # Choose best projection.
        # TODO: refactor this, enable/disable
        if self.store:
            sum_masks = np.vstack([self.store.sum_masks(cluster)
                                   for cluster in cluster_ids]).sum(axis=0)
            # Take the best 3 channels.
            channels = np.argsort(sum_masks)[::-1][:3]
        else:
            channels = np.arange(len(self.model.channels[:3]))
        self.view.dimensions = ['time'] + [(ch, 0) for ch in channels]


class CorrelogramViewModel(BaseViewModel):
    _view_class = CorrelogramView
    _view_name = 'correlograms'

    binsize = None
    winsize_bins = None

    def on_select(self, cluster_ids):
        super(CorrelogramViewModel, self).on_select(cluster_ids)
        spikes = self.spike_ids

        self.view.cluster_ids = cluster_ids

        # Compute the correlograms.
        spike_samples = self.model.spike_samples[spikes]
        spike_clusters = self.view.visual.spike_clusters
        ccgs = correlograms(spike_samples,
                            spike_clusters,
                            cluster_order=cluster_ids,
                            binsize=self.binsize,
                            winsize_bins=self.winsize_bins,
                            )
        ccgs = _symmetrize_correlograms(ccgs)
        # Normalize the CCGs.
        ccgs = ccgs * (1. / float(ccgs.max()))
        self.view.visual.correlograms = ccgs

        # Take the cluster order into account.
        self.view.visual.cluster_order = cluster_ids

    def on_cluster(self, up=None):
        super(CorrelogramViewModel, self).on_cluster(up)
        if up is None or up.description not in ('merge', 'assign'):
            return

        # TODO OPTIM: add the CCGs of the merged clusters
        # if up.description == 'merge':
        #     self.view.visual.cluster_ids = up.added
        #     n = len(up.added)
        #     self.view.visual.cluster_colors = _selected_clusters_colors(n)

        # Recompute the CCGs with the already-selected spikes, and the
        # newly-created clusters.
        if self.spike_ids is not None:
            self.on_select(up.added)


class TraceViewModel(BaseViewModel):
    _view_class = TraceView
    _view_name = 'traces'
    _interval = None

    scale_factor = 1.
    n_samples_per_spike = 20
    interval_size = .25  # default interval size in milliseconds

    def _load_traces(self, interval):
        start, end = interval
        spikes = self.spike_ids

        # Load the traces.
        debug("Loading traces...")
        # Using channel_order ensures that we get rid of the dead channels.
        # We also keep the channel order as specified by the PRM file.
        traces = self.model.traces[start:end, self.model.channel_order]
        debug("Done!")

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
        # We update the spike clusters and cluster colors according to the
        # subselection of spikes.
        self._update_spike_clusters(spikes)

        # Set the spike samples.
        spike_samples = self.model.spike_samples[spikes]
        # This is in unit of samples relative to the start of the interval.
        spike_samples = spike_samples - start
        self.view.visual.spike_samples = spike_samples
        self.view.visual.offset = start

        # Load the masks.
        masks = self._model.masks[spikes]
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

    def move(self, amount):
        """Move the current interval by a given amount (in samples)."""
        start, end = self.interval
        self.interval = start + amount, end + amount

    def move_right(self):
        """Move the current interval to the right."""
        start, end = self.interval
        self.move(+(end - start) // 20)

    def move_left(self):
        """Move the current interval to the left."""
        start, end = self.interval
        self.move(-(end - start) // 20)

    def on_key_press(self, event):
        key = event.key
        if 'Control' in event.modifiers:
            if key == 'Left':
                self.move_left()
                self.view.update()
            elif key == 'Right':
                self.move_right()
                self.view.update()

    def on_open(self):
        super(TraceViewModel, self).on_open()
        self.view.visual.n_samples_per_spike = self.model.n_samples_waveforms
        self.view.visual.sample_rate = self.model.sample_rate

    def on_select(self, cluster_ids):
        super(TraceViewModel, self).on_select(cluster_ids)
        spikes = self.spike_ids
        # Select the default interval.
        half_size = int(self.interval_size * self.model.sample_rate / 2.)
        if len(spikes) > 0:
            # Center the default interval around the first spike.
            sample = self._model.spike_samples[spikes[0]]
        else:
            sample = half_size
        # Load traces by setting the interval.
        self.interval = sample - half_size, sample + half_size

    def on_cluster(self, up=None):
        """May be overriden."""
        self._update_spike_clusters(self.view.visual.spike_ids)
