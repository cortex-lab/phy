# -*- coding: utf-8 -*-

"""Tests of view model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ....utils.testing import show_test
from ....io.mock.artificial import MockModel
from ..clustering import Clustering
from ..views import (WaveformViewModel,
                     FeatureViewModel,
                     CorrelogramViewModel,
                     )


#------------------------------------------------------------------------------
# View model tests
#------------------------------------------------------------------------------

def _test_view_model(view_model_class):
    model = MockModel()
    clustering = Clustering(model.spike_clusters)

    clusters = [3, 4]
    spikes = clustering.spikes_in_clusters(clusters)

    vm = view_model_class(model, scale_factor=1.)
    vm.on_open()
    vm.on_select(clusters, spikes)

    show_test(vm.view)


def test_waveforms():
    _test_view_model(WaveformViewModel)


def test_features():
    _test_view_model(FeatureViewModel)


def test_ccg():
    _test_view_model(CorrelogramViewModel)
