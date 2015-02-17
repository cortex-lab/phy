# -*- coding: utf-8 -*-

"""Manual sorting interface."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...notebook.utils import enable_notebook
from ...utils.logging import set_level, warn
from ._history import GlobalHistory
from .clustering import Clustering
from ...io.kwik_model import KwikModel
from .cluster_view import ClusterView, cluster_info
from .cluster_info import ClusterMetadata, ClusterStats
from .selector import Selector
from .session import Session
from ...io.base_model import BaseModel
from ...plot.waveforms import WaveformView
from ...notebook.utils import load_css, ipython_shell


#------------------------------------------------------------------------------
# Default interface
#------------------------------------------------------------------------------

def _mean_masks(masks, spikes):
    """Return the mean mask vector for a set of spikes."""
    return masks[spikes].mean(axis=0)


def create_clustering_session(filename=None, model=None, backend=None):
    """Create a manual clustering session in the IPython notebook.

    Parameters
    ----------
    filename : str
        Path to a .kwik file, to be used if 'model' is not used.
    model : instance of BaseModel
        A Model instance, to be used if 'filename' is not used.

    """

    session = Session()
    session.model = None

    # Public actions
    # -------------------------------------------------------------------------

    @session.action(title='Open')
    def open(filename=None, model=None):
        if model is None:
            model = KwikModel(filename)
        session.model = model
        session.emit('open')

    @session.action(title='Select clusters')
    def select(clusters):
        session.selector.selected_clusters = clusters
        session.emit('select')

    @session.action(title='Merge')
    def merge(clusters):
        up = session.clustering.merge(clusters)
        session.emit('cluster', up=up)

    @session.action(title='Split')
    def split(spikes):
        up = session.clustering.split(spikes)
        session.emit('cluster', up=up)

    @session.action(title='Move clusters to a group')
    def move(clusters, group):
        up = session.cluster_metadata.set_group(clusters, group)
        session.emit('cluster', up=up)

    @session.action(title='Undo')
    def undo():
        up = session._global_history.undo()
        session.emit('cluster', up=up, add_to_stack=False)

    @session.action(title='Redo')
    def redo():
        up = session._global_history.redo()
        session.emit('cluster', up=up, add_to_stack=False)

    # Event callbacks
    # -------------------------------------------------------------------------

    @session.connect
    def on_open():
        """Update the session after new data has been loaded."""
        session._global_history = GlobalHistory()
        # TODO: call this after the channel groups has changed.
        # Update the Selector and Clustering instances using the Model.
        spike_clusters = session.model.spike_clusters
        session.clustering = Clustering(spike_clusters)
        session.cluster_metadata = session.model.cluster_metadata
        session.stats = ClusterStats()
        # TODO: n_spikes_max in a user parameter
        session.selector = Selector(spike_clusters, n_spikes_max=100)
        # TODO: user-customizable list of statistics

        mask_selector = Selector(spike_clusters, n_spikes_max=100)

        @session.stats.stat
        def cluster_masks(cluster):
            mask_selector.selected_clusters = [cluster]
            spikes = mask_selector.selected_spikes
            return _mean_masks(session.model.masks, spikes)

    @session.connect
    def on_cluster(up=None, add_to_stack=True):
        if add_to_stack:
            session._global_history.action(session.clustering)
            # TODO: if metadata
            # session._global_history.action(session.cluster_metadata)

    # Views
    # -------------------------------------------------------------------------

    @session.action(title='Show waveforms')
    def show_waveforms():
        if backend in ('pyqt4', None):
            kwargs = {'always_on_top': True}
        else:
            kwargs = {}
        view = WaveformView(**kwargs)

        @session.connect
        def on_open():
            if session.model is None:
                return
            view.visual.spike_clusters = session.clustering.spike_clusters
            view.visual.cluster_metadata = session.cluster_metadata
            view.visual.channel_positions = session.model.probe.positions
            view.update()

        @session.connect
        def on_cluster(up=None):
            pass
            # TODO: select the merged cluster
            # session.select(merged)

        @session.connect
        def on_select():
            spikes = session.selector.selected_spikes
            if len(spikes) == 0:
                return
            view.visual.waveforms = session.model.waveforms[spikes]
            view.visual.masks = session.model.masks[spikes]
            view.visual.spike_labels = spikes
            view.update()

        # Unregister the callbacks when the view is closed.
        @view.connect
        def on_close(event):
            session.unconnect(on_open, on_cluster, on_select)

        view.show()

        # Update the view if the model was already opened.
        on_open()
        on_select()

        return view

    @session.action(title='Show clusters')
    def show_clusters():
        """Create and show a new cluster view."""

        cluster_colors = [session.cluster_metadata.color(cluster)
                          for cluster in session.clustering.cluster_labels]
        try:
            clusters = [ cluster_info(c, quality=0, nchannels=1, nspikes=2, ccg=None) for c in session.clustering.cluster_labels]
            view = ClusterView(clusters=clusters, colors=cluster_colors)
        except RuntimeError:
            warn("The cluster view only works in IPython.")
            return
        view.on_trait_change(lambda _, __, clusters: session.select(clusters),
                             'value')
        load_css('static/d3clusterwidget.css')
        load_css('static/widgets.css')
        session.view = view
        from IPython.display import display
        display(view)
        return view

    return session


def start_manual_clustering(filename=None, model=None, session=None,
                            backend=None):
    """Start a manual clustering session in the IPython notebook.

    Parameters
    ----------
    session : Session
        A Session instance
    filename : str
        Path to a .kwik file, to be used if 'model' is not used.
    model : instance of BaseModel
        A Model instance, to be used if 'filename' is not used.

    """

    if session is None:
        session = create_clustering_session(filename=filename, model=model,
                                            backend=backend)

    # Enable the notebook interface.
    enable_notebook(backend=backend)

    session.open(filename=filename, model=model)
    session.show_clusters()

    return session
