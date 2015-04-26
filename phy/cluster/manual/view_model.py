# -*- coding: utf-8 -*-

"""Views for Kwik model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.logging import debug
from ...plot.ccg import CorrelogramView
from ...plot.features import FeatureView
from ...plot.waveforms import WaveformView
from ...plot.traces import TraceView
from ...stats.ccg import correlograms, _symmetrize_correlograms


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

    def __init__(self, model, store=None, **kwargs):
        self._model = model
        self._store = store
        # Selected spikes.
        self._spikes = None
        self._cluster_ids = None
        for key, value in kwargs.items():
            setattr(self, key, value)
        vispy_kwargs_names = ('position', 'size',)
        vispy_kwargs = {name: kwargs[name] for name in vispy_kwargs_names
                        if name in kwargs}
        backend = kwargs.pop('backend', None)
        self._view = _create_view(self._view_class,
                                  backend=backend,
                                  **vispy_kwargs)

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
    def view(self):
        return self._view

    def _load_from_store_or_model(self, name, cluster_ids, spikes):
        if self._store is not None:
            return self._store.load(name, cluster_ids, spikes)
        else:
            return getattr(self._model, name)[spikes]

    def _update_cluster_colors(self):
        n = self.view.visual.n_clusters
        self.view.visual.cluster_colors = _selected_clusters_colors(n)

    def _update_spike_clusters(self, spikes):
        assert spikes is not None
        self._view.visual.spike_clusters = self.model.spike_clusters[spikes]

    def on_open(self):
        """To be overriden."""

    def on_cluster(self, up=None):
        """May be overriden."""
        self._update_spike_clusters(self._spikes)
        self._update_cluster_colors()

    def on_select(self, cluster_ids, spikes):
        """To be overriden."""
        self._spikes = spikes
        self._cluster_ids = cluster_ids
        self._update_spike_clusters(spikes)
        self._update_cluster_colors()

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

    def on_select(self, cluster_ids, spikes):
        super(WaveformViewModel, self).on_select(cluster_ids, spikes)

        # Load waveforms.
        debug("Loading {0:d} waveforms...".format(len(spikes)))
        waveforms = self.model.waveforms[spikes]
        debug("Done!")

        # Spikes.
        self.view.visual.spike_ids = spikes

        waveforms *= self.scale_factor
        self.view.visual.waveforms = waveforms

        # Load masks.
        masks = self._load_from_store_or_model('masks',
                                               cluster_ids,
                                               spikes)
        self.view.visual.masks = masks

    def on_close(self):
        self.view.visual.spike_clusters = []
        self.view.visual.channel_positions = []
        self.view.update()


class FeatureViewModel(BaseViewModel):
    _view_class = FeatureView
    _view_name = 'features'
    scale_factor = 1.

    def on_select(self, cluster_ids, spikes):
        super(FeatureViewModel, self).on_select(cluster_ids, spikes)

        # Spikes.
        self.view.visual.spike_ids = spikes
        self.view.visual.spike_samples = self.model.spike_samples[spikes]

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

    def on_select(self, cluster_ids, spikes):
        super(CorrelogramViewModel, self).on_select(cluster_ids, spikes)
        self.view.cluster_ids = cluster_ids
        spike_clusters = self.model.spike_clusters[spikes]
        spike_samples = self.model.spike_samples[spikes]

        # Compute the correlograms.
        ccgs = correlograms(spike_samples,
                            spike_clusters,
                            binsize=self.binsize,
                            winsize_bins=self.winsize_bins,
                            )
        ccgs = _symmetrize_correlograms(ccgs)

        # Normalize the CCGs.
        ccgs = ccgs * (1. / float(ccgs.max()))
        self.view.visual.correlograms = ccgs

        # Cluster colors.
        self._update_cluster_colors()

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
        if self._spikes is not None:
            self.on_select(up.added, self._spikes)


class TraceViewModel(BaseViewModel):
    _view_class = TraceView
    _view_name = 'traces'
    scale_factor = 1.
    _interval = None

    def _load_traces(self, interval):
        start, end = interval
        cluster_ids = self._cluster_ids

        debug("Loading traces...")
        traces = self.model.traces[start:end, :]
        debug("Done!")

        traces *= self.scale_factor
        self.view.visual.traces = traces

        # Keep the spikes in the interval.
        spike_samples = self.model.spike_samples[self._spikes]
        a, b = spike_samples.searchsorted(interval)
        spikes = self._spikes[a:b]
        spike_samples = self.model.spike_samples[spikes]
        self._update_spike_clusters(spikes)
        spike_samples -= start
        self.view.visual.n_spikes = len(spikes)
        self.view.visual.spike_ids = spikes
        self.view.visual.spike_samples = spike_samples

        # Load masks.
        masks = self._load_from_store_or_model('masks', cluster_ids, spikes)
        self.view.visual.masks = masks

        # TODO
        self.view.visual.offset = 0

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("The interval should be a (start, end) tuple.")
        self._interval = value
        self._load_traces(value)
        self.view.update()

    def move(self, amount):
        start, end = self.interval
        self.interval = start + amount, end + amount

    def move_right(self):
        start, end = self.interval
        self.move((end - start) // 2)

    def move_left(self):
        start, end = self.interval
        self.move(-(end - start) // 2)

    def on_open(self):
        super(TraceViewModel, self).on_open()
        self.view.visual.n_samples_per_spike = 20
        self.view.visual.sample_rate = 20000.

    def on_select(self, cluster_ids, spikes):
        self._spikes = spikes
        self._cluster_ids = cluster_ids

        self._update_spike_clusters(spikes)

        # Load traces.
        # TODO: select the default interval
        # TODO: x-scale according to the interval, not to the number of shown
        # samples
        self.interval = (2000, 5000)

        n = self.view.visual.n_clusters
        self.view.visual.cluster_colors = _selected_clusters_colors(n)

    def on_close(self):
        self.view.visual.spike_clusters = []
        self.view.update()
