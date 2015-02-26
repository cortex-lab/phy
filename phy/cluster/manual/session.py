# -*- coding: utf-8 -*-

"""Session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from functools import partial
import shutil

import numpy as np

from ...ext.six import string_types
from ...utils._misc import (_phy_user_dir,
                            _ensure_phy_user_dir_exists)
from ...ext.slugify import slugify
from ...utils.event import EventEmitter
from ...notebook.utils import enable_notebook
from ...utils.logging import set_level, warn
from ._history import GlobalHistory
from .clustering import Clustering
from ...io.kwik_model import KwikModel
from ...notebook.utils import load_css, ipython_shell
from ...notebook.cluster_view import ClusterView
from .cluster_info import ClusterMetadata
from .store import ClusterStore
from .selector import Selector
from ...io.base_model import BaseModel
from ...plot.waveforms import WaveformView


#------------------------------------------------------------------------------
# BaseSession class
#------------------------------------------------------------------------------

class BaseSession(EventEmitter):
    """Provide actions, views, and an event system for creating an interactive
    session."""
    def __init__(self):
        super(BaseSession, self).__init__()
        self._actions = []

    def action(self, func=None, title=None):
        """Decorator for a callback function of an action.

        The 'title' argument is used as a title for the GUI button.

        """
        if func is None:
            return partial(self.action, title=title)

        # HACK: handle the case where the first argument is the title.
        if isinstance(func, string_types):
            return partial(self.action, title=func)

        # Register the action.
        self._actions.append({'func': func, 'title': title})

        # Set the action function as a Session method.
        setattr(self, func.__name__, func)

        return func


#------------------------------------------------------------------------------
# Session class
#------------------------------------------------------------------------------

def _ensure_disk_store_exists(dir_name, root_path=None):
    # Disk store.
    if root_path is None:
        _ensure_phy_user_dir_exists()
        root_path = _phy_user_dir('cluster_store')
    # Create the disk store if it does not exist.
    if not op.exists(root_path):
        os.mkdir(root_path)
    # Put the store in a subfolder, using the name.
    dir_name = slugify(dir_name)
    path = op.join(root_path, dir_name)
    if not op.exists(path):
        os.mkdir(path)
    return path


class Session(BaseSession):
    """Default manual clustering session in the IPython notebook.

    Parameters
    ----------
    filename : str
        Path to a .kwik file, to be used if 'model' is not used.
    model : instance of BaseModel
        A Model instance, to be used if 'filename' is not used.
    backend : str
        VisPy backend. For example 'pyqt4' or 'ipynb_webgl'.

    """
    def __init__(self, store_path=None, backend=None):
        super(Session, self).__init__()
        self.model = None
        self._backend = backend
        self._store_path = store_path

        # self.action and self.connect are decorators.
        self.action(self.open, title='Open')
        self.action(self.select, title='Select clusters')
        self.action(self.merge, title='Merge')
        self.action(self.split, title='Split')
        self.action(self.move, title='Move clusters to a group')
        self.action(self.undo, title='Undo')
        self.action(self.redo, title='Redo')
        self.action(self.show_waveforms, title='Show waveforms')
        self.action(self.show_clusters, title='Show clusters')

        self.connect(self.on_open)
        self.connect(self.on_cluster)

    # Public actions
    # -------------------------------------------------------------------------

    def open(self, filename=None, model=None):
        if model is None:
            model = KwikModel(filename)
        self.model = model
        self.emit('open')

    def select(self, clusters):
        self.selector.selected_clusters = clusters
        self.emit('select')

    def merge(self, clusters):
        up = self.clustering.merge(clusters)
        self.emit('cluster', up=up)

    def split(self, spikes):
        up = self.clustering.split(spikes)
        self.emit('cluster', up=up)

    def move(self, clusters, group):
        up = self.cluster_metadata.set_group(clusters, group)
        self.emit('cluster', up=up)

    def undo(self):
        up = self._global_history.undo()
        self.emit('cluster', up=up, add_to_stack=False)

    def redo(self):
        up = self._global_history.redo()
        self.emit('cluster', up=up, add_to_stack=False)

    # Event callbacks
    # -------------------------------------------------------------------------

    def on_open(self):
        """Update the session after new data has been loaded."""
        self._global_history = GlobalHistory()
        # TODO: call this after the channel groups has changed.
        # Update the Selector and Clustering instances using the Model.
        spike_clusters = self.model.spike_clusters
        self.clustering = Clustering(spike_clusters)
        self.cluster_metadata = self.model.cluster_metadata
        # TODO: n_spikes_max in a user parameter
        self.selector = Selector(spike_clusters, n_spikes_max=100)

        path = _ensure_disk_store_exists(self.model.name,
                                         root_path=self._store_path)
        self.store = ClusterStore(path)
        # TODO: fill the store

        @self.connect
        def on_cluster(up=None, add_to_stack=True):
            # TODO: Update the store
            pass

        # mask_selector = Selector(spike_clusters, n_spikes_max=100)
        # @self.stats.stat
        # def cluster_masks(cluster):
        #     mask_selector.selected_clusters = [cluster]
        #     spikes = mask_selector.selected_spikes
        #     return self.model.masks[spikes].mean(axis=0)

    def on_cluster(self, up=None, add_to_stack=True):
        if add_to_stack:
            self._global_history.action(self.clustering)
            # TODO: if metadata
            # self._global_history.action(self.cluster_metadata)

    # Views
    # -------------------------------------------------------------------------

    def show_waveforms(self):
        if self._backend in ('pyqt4', None):
            kwargs = {'always_on_top': True}
        else:
            kwargs = {}
        view = WaveformView(**kwargs)

        @self.connect
        def on_open():
            if self.model is None:
                return
            view.visual.spike_clusters = self.clustering.spike_clusters
            view.visual.cluster_metadata = self.cluster_metadata
            view.visual.channel_positions = self.model.probe.positions
            view.update()

        @self.connect
        def on_cluster(up=None):
            pass
            # TODO: select the merged cluster
            # self.select(merged)

        @self.connect
        def on_select():
            spikes = self.selector.selected_spikes
            if len(spikes) == 0:
                return
            view.visual.waveforms = self.model.waveforms[spikes]
            view.visual.masks = self.model.masks[spikes]
            view.visual.spike_ids = spikes
            view.update()

        # Unregister the callbacks when the view is closed.
        @view.connect
        def on_close(event):
            self.unconnect(on_open, on_cluster, on_select)

        view.show()

        # Update the view if the model was already opened.
        on_open()
        on_select()

        return view

    def show_clusters(self):
        """Create and show a new cluster view."""

        # TODO: no more 1 cluster = 1 color, use a fixed set of colors
        # for the selected clusters.
        cluster_colors = [self.cluster_metadata.color(cluster)
                          for cluster in self.clustering.cluster_ids]
        try:
            view = ClusterView(clusters=self.clustering.cluster_ids,
                               colors=cluster_colors)
        except RuntimeError:
            warn("The cluster view only works in IPython.")
            return
        view.on_trait_change(lambda _, __, clusters: self.select(clusters),
                             'value')
        load_css('static/widgets.css')
        from IPython.display import display
        display(view)
        return view


#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

def start_manual_clustering(filename=None, model=None, session=None,
                            store_path=None, backend=None):
    """Start a manual clustering session in the IPython notebook.

    Parameters
    ----------
    session : BaseSession
        A BaseSession instance
    filename : str
        Path to a .kwik file, to be used if 'model' is not used.
    model : instance of BaseModel
        A Model instance, to be used if 'filename' is not used.

    """

    if session is None:
        session = Session(store_path=store_path, backend=backend)

    # Enable the notebook interface.
    enable_notebook(backend=backend)

    session.open(filename=filename, model=model)
    session.show_clusters()

    return session
