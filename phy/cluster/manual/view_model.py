# -*- coding: utf-8 -*-

"""Views for Kwik model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.array import get_excerpts
from ...utils.logging import debug
from ...plot.ccg import CorrelogramView
from ...plot.features import FeatureView
from ...plot.waveforms import WaveformView
from ...stats.ccg import correlograms, _symmetrize_correlograms


#------------------------------------------------------------------------------
# BaseViewModel for plot views and Kwik model
#------------------------------------------------------------------------------

def _create_view(cls, backend=None):
    if backend in ('pyqt4', None):
        kwargs = {'always_on_top': True}
    else:
        kwargs = {}
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

    def __init__(self, model, store=None, backend=None, scale_factor=1.):
        self._model = model
        self._store = store
        self._backend = backend
        self._scale_factor = scale_factor
        self._view = _create_view(self._view_class, backend=backend)

    @property
    def model(self):
        return self._model

    @property
    def store(self):
        return self._store

    @property
    def view(self):
        return self._view

    def _load_from_store_or_model(self, name, clusters, spikes):
        if self._store is not None:
            return self._store.load(name, clusters, spikes)
        else:
            return getattr(self._model, name)[spikes]

    def on_open(self):
        """To be overriden."""
        self.view.visual.spike_clusters = self.model.spike_clusters

    def on_cluster(self, up):
        """To be overriden."""
        pass

    def on_select(self, clusters, spikes):
        """To be overriden."""
        pass

    def show(self):
        # self._view.update()
        self._view.show()


#------------------------------------------------------------------------------
# View models
#------------------------------------------------------------------------------

class WaveformViewModel(BaseViewModel):
    _view_class = WaveformView

    def on_open(self):
        self.view.visual.spike_clusters = self.model.spike_clusters
        self.view.visual.channel_positions = self.model.probe.positions

    def on_select(self, clusters, spikes):
        # Load waveforms.
        debug("Loading {0:d} waveforms...".format(len(spikes)))
        waveforms = self.model.waveforms[spikes]
        debug("Done!")

        waveforms *= self._scale_factor
        self.view.visual.waveforms = waveforms

        # Load masks.
        masks = self._load_from_store_or_model('masks',
                                               clusters,
                                               spikes)
        self.view.visual.masks = masks

        # Spike ids.
        self.view.visual.spike_ids = spikes

        # Cluster colors.
        n = len(clusters)
        self.view.visual.cluster_colors = _selected_clusters_colors(n)

    def on_close(self):
        self.view.visual.spike_clusters = []
        self.view.visual.channel_positions = []
        self.view.update()


class FeatureViewModel(BaseViewModel):
    _view_class = FeatureView

    def on_select(self, clusters, spikes):
        # Load features.
        features = self._load_from_store_or_model('features',
                                                  clusters,
                                                  spikes)
        # Load masks.
        masks = self._load_from_store_or_model('masks',
                                               clusters,
                                               spikes)

        # WARNING: convert features to a 3D array
        # (n_spikes, n_channels, n_features)
        # because that's what the FeatureView expects currently.
        n_fet = self.model.n_features_per_channel
        n_channels = self.model.n_channels
        shape = (-1, n_channels, n_fet)
        features = features[:, :n_fet * n_channels].reshape(shape)
        # Scale factor.
        features *= self._scale_factor

        self.view.visual.features = features
        self.view.visual.masks = masks

        # Choose best projection.
        # TODO: refactor this, enable/disable
        if self.store:
            sum_masks = np.vstack([self.store.sum_masks(cluster)
                                   for cluster in clusters]).sum(axis=0)
            # Take the best 3 channels.
            channels = np.argsort(sum_masks)[::-1][:3]
        else:
            channels = np.arange(len(self.model.channels[:3]))
        self.view.visual.dimensions = [(ch, 0) for ch in channels]

        # *All* spike clusters.
        self.view.visual.spike_clusters = self.model.spike_clusters

        # Spike times and ids.
        self.view.visual.spike_times = self.model.spike_times[spikes]
        self.view.visual.spike_ids = spikes

        # Cluster colors.
        n = len(clusters)
        self.view.visual.cluster_colors = _selected_clusters_colors(n)


class CorrelogramViewModel(BaseViewModel):
    _view_class = CorrelogramView

    def on_select(self, clusters, spikes):
        self.view.visual.clusters_ids = clusters

        def _extract(arr):
            # TODO: user-definable CCG parameters
            return get_excerpts(arr, n_excerpts=100, excerpt_size=100)

        # Extract a subset of the spikes belonging to the selected clusters.
        spikes_subset = _extract(spikes)
        spike_clusters = self.model.spike_clusters[spikes_subset]
        spike_times = self.model.spike_times[spikes_subset]

        # Compute the correlograms.
        ccgs = correlograms(spike_times, spike_clusters,
                            binsize=20, winsize_bins=51)
        ccgs = _symmetrize_correlograms(ccgs)

        # Normalize the CCGs.
        ccgs = ccgs * (1. / float(ccgs.max()))
        self.view.visual.correlograms = ccgs

        # Cluster colors.
        n = len(clusters)
        self.view.visual.cluster_colors = _selected_clusters_colors(n)
