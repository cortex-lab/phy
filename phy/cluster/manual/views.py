# -*- coding: utf-8 -*-

"""Views for Kwik model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..utils._color import _random_color
from ...plot.ccgs import CorrelogramView
from ...plot.features import FeatureView
from ...plot.waveforms import WaveformView


#------------------------------------------------------------------------------
# ViewModel for plot views and Kwik model
#------------------------------------------------------------------------------

def _create_view(cls, backend=None):
    if backend in ('pyqt4', None):
        kwargs = {'always_on_top': True}
    else:
        kwargs = {}
    return cls(**kwargs)


class BaseViewModel(object):
    """Used to create views from a model."""
    _view_class = None

    def __init__(self, model, backend=None):
        self._model = model
        self._backend = backend
        self._view = _create_view(self._view_class, backend=backend)

    @property
    def model(self):
        return self._model

    @property
    def view(self):
        return self._view

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


class WaveformViewModel(BaseViewModel):
    _view_class = WaveformView

    def on_open(self):
        self.view.visual.spike_clusters = self.model.spike_clusters
        self.view.visual.channel_positions = self.model.probe.positions

    def on_select(self, clusters, spikes):
        self.view.visual.waveforms = self.model.waveforms[spikes]
        self.view.visual.masks = self.model.masks[spikes]
        self.view.visual.spike_ids = spikes
        # TODO: how to choose cluster colors?
        self.view.visual.cluster_colors = [_random_color() for _ in clusters]


class FeatureViewModel(BaseViewModel):
    _view_class = FeatureView

    def on_select(self, clusters, spikes):
        self.view.visual.features = self.model.features[spikes]
        self.view.visual.masks = self.model.masks[spikes]
        self.view.visual.spike_ids = spikes
        self.view.visual.cluster_colors = [_random_color() for _ in clusters]


class CorrelogramViewModel(BaseViewModel):
    _view_class = CorrelogramView

    def on_select(self, clusters, spikes):
        # TODO: correlograms
        self.view.visual.correlograms = correlograms
        self.view.visual.clusters_ids = clusters
        self.view.visual.cluster_colors = [_random_color() for _ in clusters]
