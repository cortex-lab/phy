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
from .cluster_view import ClusterView
from .cluster_info import ClusterMetadata
from .selector import Selector
from .session import Session
from ...io.base_model import BaseModel
from ...plot.waveforms import WaveformView
from ...notebook.utils import load_css, ipython_shell


#------------------------------------------------------------------------------
# Default interface
#------------------------------------------------------------------------------

def start_manual_clustering(filename=None, model=None):
    """Start a manual clustering session in the IPython notebook.

    Parameters
    ----------
    filename : str
        Path to a .kwik file, to be used if 'model' is not used.
    model : instance of BaseModel
        A Model instance, to be used if 'filename' is not used.

    """

    session = Session()

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
        session.emit('select', clusters=clusters)

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
        up = session.cluster_metadata.set(clusters, 'group', group)
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
        # TODO: cluster stats
        # TODO: n_spikes_max in a user parameter
        session.selector = Selector(spike_clusters, n_spikes_max=100)

    @session.connect
    def on_select(clusters=None):
        if clusters is None:
            clusters = session.selected_clusters

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
        view = WaveformView()

        @session.connect
        def on_open():
            view.visual.spike_clusters = session.clustering.spike_clusters
            view.visual.cluster_metadata = session.cluster_metadata
            view.visual.channel_positions = session.model.probe.positions
            view.update()

        @session.connect
        def on_cluster(up=None, add_to_stack=None):
            pass
            # TODO: select the merged cluster
            # session.select(merged)

        @session.connect
        def on_select(clusters):
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
        return view

    @session.action(title='Show clusters')
    def show_clusters():
        """Create and show a new cluster view."""

        cluster_colors = [session.cluster_metadata[cluster]['color']
                          for cluster in session.clustering.cluster_labels]

        try:
            view = ClusterView(clusters=session.clustering.cluster_labels,
                               colors=cluster_colors)
        except RuntimeError:
            warn("The cluster view only works in IPython.")
            return
        view.on_trait_change(lambda _, __, clusters: session.select(clusters),
                             'value')
        load_css('static/widgets.css')
        from IPython.display import display
        display(view)
        return view

    # Enable the notebook interface.
    enable_notebook()

    session.open(filename=filename, model=model)
    session.show_clusters()

    return session
