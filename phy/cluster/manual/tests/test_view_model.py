# -*- coding: utf-8 -*-

"""Tests of view model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ....utils.testing import (show_test_start,
                               show_test_stop,
                               show_test_run,
                               )
from ....io.mock.artificial import MockModel
from ..clustering import Clustering
from ..view_model import (WaveformViewModel,
                          FeatureViewModel,
                          CorrelogramViewModel,
                          TraceViewModel,
                          )


#------------------------------------------------------------------------------
# View model tests
#------------------------------------------------------------------------------

def _test_view_model(view_model_class, **kwargs):
    n_frames = 20

    model = MockModel()
    clustering = Clustering(model.spike_clusters)

    clusters = [3, 4]
    spikes = clustering.spikes_in_clusters(clusters)

    vm = view_model_class(model, **kwargs)
    vm.on_open()
    vm.on_select(clusters, spikes)

    # Show the view.
    show_test_start(vm.view)
    show_test_run(vm.view, n_frames)

    # Merge the clusters and update the view.
    up = clustering.merge(clusters)
    vm.on_cluster(up)
    show_test_run(vm.view, n_frames)

    # Split some spikes and update the view.
    spikes = spikes[::2]
    up = clustering.assign(spikes, np.random.randint(low=0,
                                                     high=5,
                                                     size=len(spikes)))
    vm.on_cluster(up)
    show_test_run(vm.view, n_frames)

    show_test_stop(vm.view)

    return vm


def test_waveforms():
    _test_view_model(WaveformViewModel)


def test_features():
    _test_view_model(FeatureViewModel)


def test_ccg():
    _test_view_model(CorrelogramViewModel,
                     binsize=20,
                     winsize_bins=51,
                     n_excerpts=100,
                     excerpt_size=100,
                     )


def test_traces():
    _test_view_model(TraceViewModel)
