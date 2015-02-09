# -*- coding: utf-8 -*-

"""Manual sorting interface."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from IPython.display import display

from ...notebook.utils import enable_notebook
from ...utils.logging import set_level, warn
from ._history import GlobalHistory
from .clustering import Clustering
from ...io.kwik_model import KwikModel
from .cluster_view import ClusterView
from .cluster_metadata import ClusterMetadata
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

    # Enable the notebook interface.
    enable_notebook()

    if model is None:
        model = KwikModel(filename)

    session = Session(model)

    @session.create("Show waveforms")
    def show_waveforms():
        view = WaveformView()
        view.show()
        return view

    @session.callback(WaveformView)
    def on_load(view):
        view.visual.spike_clusters = session.clustering.spike_clusters
        view.visual.cluster_metadata = session.cluster_metadata
        view.visual.channel_positions = session.model.probe.positions

    @session.callback(WaveformView)
    def on_select(view):
        spikes = session.selector.selected_spikes
        view.visual.waveforms = session.model.waveforms[spikes]
        view.visual.masks = session.model.masks[spikes]
        view.visual.spike_labels = spikes

    @session.callback(WaveformView)
    def on_cluster(view, up=None):
        # TODO
        pass

    @session.create("Show clusters")
    def show_clusters():
        """Create and show a new cluster view."""
        try:
            view = ClusterView(clusters=session.cluster_labels,
                               colors=session.cluster_colors)
        except RuntimeError:
            warn("The cluster view only works in IPython.")
            return
        view.on_trait_change(lambda _, __, clusters: session.select(clusters),
                             'value')
        load_css('static/widgets.css')
        display(view)
        return view

    session.show_clusters()

    return session
