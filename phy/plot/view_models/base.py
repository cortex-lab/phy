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

    def show(self):
        self._view.show()
