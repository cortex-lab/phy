# -*- coding: utf-8 -*-

"""Manual sorting interface."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...notebook.utils import enable_notebook
from ...utils.logging import set_level
from ._history import GlobalHistory
from .clustering import Clustering
from ...io.kwik_model import KwikModel
from .cluster_view import ClusterView
from .cluster_metadata import ClusterMetadata
from .selector import Selector
from .session import Session
from ...io.base_model import BaseModel
from ...plot.waveforms import WaveformView
from ...notebook.utils import load_css


#------------------------------------------------------------------------------
# Default interface
#------------------------------------------------------------------------------

def start_manual_clustering(filename):
    """Start a manual clustering session in the IPython notebook.

    Parameters
    ----------
    filename : str
        Path to a .kwik file.

    """
    enable_notebook()
    experiment = KwikModel(filename)
    session = Session(experiment)

    @session.views.create("Show waveforms")
    def show_waveforms():
        view = WaveformView()
        view.show()

    @session.views.select
    def update_waveforms_after_select(view):
        if isinstance(view, WaveformView):
            spikes = session.selector.selected_spikes
            view.visual.waveforms = session.model.waveforms[spikes]
            view.visual.masks = session.model.masks[spikes]
            view.visual.spike_labels = spikes

    @session.views.cluster
    def update_waveforms_after_cluster(view, up=None):
        if isinstance(view, WaveformView):
            pass

    session.show_clusters()
    session.show_waveforms()
