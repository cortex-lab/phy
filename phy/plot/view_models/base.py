# -*- coding: utf-8 -*-

"""Base view model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.array import _unique
from ...utils.selector import Selector


#------------------------------------------------------------------------------
# Misc
#------------------------------------------------------------------------------

# Default color map for the selected clusters.
_COLORMAP = np.array([[102, 194, 165],
                      [252, 141, 98],
                      [141, 160, 203],
                      [231, 138, 195],
                      [166, 216, 84],
                      [255, 217, 47],
                      [229, 196, 148],
                      ])


def _create_view(cls, backend=None, **kwargs):
    if backend in ('pyqt4', None):
        kwargs.update({'always_on_top': True})
    return cls(**kwargs)


def _selected_clusters_colors(n_clusters):
    if n_clusters > _COLORMAP.shape[0]:
        colors = np.tile(_COLORMAP, (1 + n_clusters // _COLORMAP.shape[0], 1))
    else:
        colors = _COLORMAP
    return colors[:n_clusters, ...] / 255.


#------------------------------------------------------------------------------
# BaseViewModel
#------------------------------------------------------------------------------

class BaseViewModel(object):
    """Used to create views from a model."""
    _view_class = None
    _view_name = ''
    imported_params = ('n_spikes_max', 'excerpt_size')

    def __init__(self, model=None, store=None,
                 n_spikes_max=None, excerpt_size=None,
                 position=None, size=None, backend=None,
                 **kwargs):
        self._model = model
        self._store = store

        # Create the spike/cluster selector.
        self._selector = Selector(model.spike_clusters,
                                  n_spikes_max=n_spikes_max,
                                  excerpt_size=excerpt_size,
                                  )

        # Set passed keyword arguments as attributes.
        for key in self.imported_params:
            setattr(self, key, kwargs.pop(key, None))

        # Create the VisPy canvas.
        self._view = _create_view(self._view_class,
                                  backend=backend,
                                  position=position or (200, 200),
                                  size=size or (600, 600),
                                  )

        @self._view.connect
        def on_draw(event):
            if self._view.visual.empty:
                self.on_open()
                if self.cluster_ids:
                    self.on_select(self.cluster_ids)

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

    def _load_from_store_or_model(self,
                                  name,
                                  cluster_ids,
                                  spikes=None,
                                  ):
        if self._store is not None:
            return self._store.load(name,
                                    cluster_ids,
                                    spikes=spikes,
                                    )
        else:
            out = getattr(self._model, name)
            if spikes is not None:
                return out[spikes]
            else:
                return out

    def _update_spike_clusters(self, spikes=None):
        """Update the spike clusters and cluster colors."""
        if spikes is None:
            spikes = self.spike_ids
        spike_clusters = self.model.spike_clusters[spikes]
        n_clusters = len(_unique(spike_clusters))
        visual = self._view.visual
        # This updates the list of unique clusters in the view.
        visual.spike_clusters = spike_clusters
        visual.cluster_colors = _selected_clusters_colors(n_clusters)

    def on_open(self):
        """May be overriden."""

    def on_select(self, cluster_ids):
        """Must be overriden."""
        self._selector.selected_clusters = cluster_ids
        self._update_spike_clusters()

    def on_close(self):
        self._view.visual.spike_clusters = []
        self._view.update()

    def exported_params(self, save_size_pos=True):
        if save_size_pos:
            return {
                'position': self._view.position,
                'size': self._view.size,
            }
        else:
            return {}

    def show(self):
        self._view.show()
