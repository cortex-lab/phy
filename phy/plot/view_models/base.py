# -*- coding: utf-8 -*-

"""Base view model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import inspect

import numpy as np

from ...utils.array import _unique
from ...utils.selector import Selector
from ...utils._misc import _show_shortcuts
from ...utils import _as_list


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
    """Interface between a view and a model."""
    _view_class = None
    _view_name = ''
    _imported_params = ('position', 'size',)
    keyboard_shortcuts = {}
    scale_factor = 1.

    def __init__(self, model=None, store=None, wizard=None,
                 cluster_ids=None, **kwargs):

        self._model = model
        assert store is not None
        self._store = store
        self._wizard = wizard
        self._cluster_ids = None

        # Instanciate the underlying view.
        self._view = self._create_view(**kwargs)

        # Set passed keyword arguments as attributes.
        for param in self.imported_params():
            value = kwargs.get(param, None)
            if value is not None:
                setattr(self, param, value)

        self.on_open()
        if cluster_ids is not None:
            self.select(_as_list(cluster_ids))

    def connect(self, func):
        pass

    @classmethod
    def imported_params(cls):
        out = ()
        for base_class in inspect.getmro(cls):
            if base_class == object:
                continue
            out += base_class._imported_params
        return out

    def _create_view(self, **kwargs):
        """Create the view with the parameters passed to the constructor.

        Must be overriden."""
        return None

    # Public properties
    #--------------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._view_name

    @property
    def store(self):
        return self._store

    @property
    def view(self):
        return self._view

    @property
    def cluster_ids(self):
        """Selected clusters."""
        return self._cluster_ids

    @property
    def n_clusters(self):
        """Number of selected clusters."""
        return len(self._cluster_ids)

    # Public methods
    #--------------------------------------------------------------------------

    def select(self, cluster_ids):
        """Select a set of clusters."""
        cluster_ids = _as_list(cluster_ids)
        self._cluster_ids = cluster_ids
        self.on_select(cluster_ids)

    def exported_params(self, save_size_pos=True):
        """Return a dictionary of variables to save when the view is closed."""
        if save_size_pos and hasattr(self._view, 'pos'):
            return {
                'position': (self._view.x(), self._view.y()),
                'size': (self._view.width(), self._view.height()),
            }
        else:
            return {}

    def show(self):
        """Show the view."""
        self._view.show()

    # Callback methods
    #--------------------------------------------------------------------------

    def on_open(self):
        """Initialize the view after the model has been loaded.

        May be overriden."""

    def on_select(self, cluster_ids):
        """Update the view after a new selection has been made.

        Must be overriden."""

    def on_cluster(self, up):
        """Called when a clustering action occurs.

        May be overriden."""

    def on_close(self):
        """Called when the model is closed.

        May be overriden."""


#------------------------------------------------------------------------------
# HTMLViewModel
#------------------------------------------------------------------------------

class HTMLViewModel(BaseViewModel):
    """Widget with custom HTML code."""

    def _create_view(self, **kwargs):
        from PyQt4.QtWebKit import QWebView
        self._html = kwargs['html']
        view = QWebView()
        return view

    def update(self):
        if hasattr(self._html, '__call__'):
            html = self._html(self._cluster_ids)
        else:
            html = self._html
        self._view.setHtml(html)

    def on_select(self, cluster_ids):
        self.update()

    def on_cluster(self, up):
        self.update()


#------------------------------------------------------------------------------
# VispyViewModel
#------------------------------------------------------------------------------

class VispyViewModel(BaseViewModel):
    """Create a VisPy view from a model.

    This object uses an internal `Selector` instance to manage spike and
    cluster selection.

    """
    _imported_params = ('n_spikes_max', 'excerpt_size')
    keyboard_shortcuts = {}
    scale_factor = 1.

    def connect(self, func):
        self._view.connect(func)

    def _create_view(self, **kwargs):
        n_spikes_max = kwargs.get('n_spikes_max', None)
        excerpt_size = kwargs.get('excerpt_size', None)
        backend = kwargs.get('backend', None)
        position = kwargs.get('position', None)
        size = kwargs.get('size', None)

        # Create the spike/cluster selector.
        self._selector = Selector(self._model.spike_clusters,
                                  n_spikes_max=n_spikes_max,
                                  excerpt_size=excerpt_size,
                                  )

        # Create the VisPy canvas.
        view = _create_view(self._view_class,
                            backend=backend,
                            position=position or (200, 200),
                            size=size or (600, 600),
                            )
        view.connect(self.on_key_press)
        return view

    @property
    def selector(self):
        return self._selector

    @property
    def cluster_ids(self):
        """Selected clusters."""
        return self._selector.selected_clusters

    @property
    def spike_ids(self):
        """Selected spikes."""
        return self._selector.selected_spikes

    @property
    def n_clusters(self):
        """Number of selected clusters."""
        return self._selector.n_clusters

    @property
    def n_spikes(self):
        """Number of selected spikes."""
        return self._selector.n_spikes

    def update_spike_clusters(self, spikes=None, spike_clusters=None):
        """Update the spike clusters and cluster colors."""
        if spikes is None:
            spikes = self.spike_ids
        if spike_clusters is None:
            spike_clusters = self.model.spike_clusters[spikes]
        n_clusters = len(_unique(spike_clusters))
        visual = self._view.visual
        # This updates the list of unique clusters in the view.
        visual.spike_clusters = spike_clusters
        visual.cluster_colors = _selected_clusters_colors(n_clusters)

    def select(self, cluster_ids):
        """Select a set of clusters."""
        self._selector.selected_clusters = cluster_ids
        self.on_select(cluster_ids)

    def on_select(self, cluster_ids):
        """Update the view after a new selection has been made.

        Must be overriden."""
        self.update_spike_clusters()
        self._view.update()

    def on_close(self):
        """Clear the view when the model is closed."""
        self._view.visual.spike_clusters = []
        self._view.update()

    def on_key_press(self, event):
        if event.key == 'h' and 'control' not in event.modifiers:
            shortcuts = self._view.keyboard_shortcuts
            shortcuts.update(self.keyboard_shortcuts)
            _show_shortcuts(shortcuts, name=self.name)

    def exported_params(self, save_size_pos=True):
        """Return a dictionary of variables to save when the view is closed."""
        if save_size_pos:
            # These fields are implemented in VisPy Canvas.
            return {
                'position': self._view.position,
                'size': self._view.size,
            }
        else:
            return {}
