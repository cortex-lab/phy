# -*- coding: utf-8 -*-

"""Tests of view model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

from pytest import mark

from ...utils.array import _spikes_per_cluster
from ...utils.logging import set_level
from ...utils.tempdir import TemporaryDirectory
from ...utils.testing import (show_test_start,
                              show_test_stop,
                              show_test_run,
                              )
from ...io.kwik.mock import create_mock_kwik
from ...io.kwik import KwikModel, create_store
from ..view_models import (WaveformViewModel,
                           FeatureGridViewModel,
                           CorrelogramViewModel,
                           TraceViewModel,
                           )


# Skip these tests in "make test-quick".
pytestmark = mark.long()


#------------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------------

_N_CLUSTERS = 5
_N_SPIKES = 200
_N_CHANNELS = 28
_N_FETS = 3
_N_SAMPLES_TRACES = 10000
_N_FRAMES = int((float(os.environ.get('PHY_EVENT_LOOP_DELAY', 0)) * 60) or 2)


def setup():
    set_level('info')


def _test_empty(view_model_class, stop=True, **kwargs):
    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=1,
                                    n_spikes=1,
                                    n_channels=_N_CHANNELS,
                                    n_features_per_channel=_N_FETS,
                                    n_samples_traces=_N_SAMPLES_TRACES)
        model = KwikModel(filename)
        spikes_per_cluster = _spikes_per_cluster(model.spike_ids,
                                                 model.spike_clusters)
        store = create_store(model,
                             path=tempdir,
                             spikes_per_cluster=spikes_per_cluster,
                             features_masks_chunk_size=10,
                             waveforms_n_spikes_max=10,
                             waveforms_excerpt_size=5,
                             )
        store.generate()

        vm = view_model_class(model=model, store=store, **kwargs)
        vm.on_open()

        # Show the view.
        show_test_start(vm.view)
        show_test_run(vm.view, _N_FRAMES)
        vm.select([0])
        show_test_run(vm.view, _N_FRAMES)
        vm.select([])
        show_test_run(vm.view, _N_FRAMES)

        if stop:
            show_test_stop(vm.view)

        return vm


def _test_view_model(view_model_class, stop=True, **kwargs):

    with TemporaryDirectory() as tempdir:
        # Create the test HDF5 file in the temporary directory.
        filename = create_mock_kwik(tempdir,
                                    n_clusters=_N_CLUSTERS,
                                    n_spikes=_N_SPIKES,
                                    n_channels=_N_CHANNELS,
                                    n_features_per_channel=_N_FETS,
                                    n_samples_traces=_N_SAMPLES_TRACES)
        model = KwikModel(filename)
        spikes_per_cluster = _spikes_per_cluster(model.spike_ids,
                                                 model.spike_clusters)
        store = create_store(model,
                             path=tempdir,
                             spikes_per_cluster=spikes_per_cluster,
                             features_masks_chunk_size=15,
                             waveforms_n_spikes_max=20,
                             waveforms_excerpt_size=5,
                             )
        store.generate()

        vm = view_model_class(model=model, store=store, **kwargs)
        vm.on_open()
        show_test_start(vm.view)

        vm.select([2])
        show_test_run(vm.view, _N_FRAMES)

        vm.select([2, 3])
        show_test_run(vm.view, _N_FRAMES)

        vm.select([3, 2])
        show_test_run(vm.view, _N_FRAMES)

        if stop:
            show_test_stop(vm.view)

        return vm


#------------------------------------------------------------------------------
# Waveforms
#------------------------------------------------------------------------------

def test_waveforms_full():
    vm = _test_view_model(WaveformViewModel, stop=False)
    vm.overlap = True
    show_test_run(vm.view, _N_FRAMES)
    vm.show_mean = True
    show_test_run(vm.view, _N_FRAMES)
    show_test_stop(vm.view)


def test_waveforms_empty():
    _test_empty(WaveformViewModel)


#------------------------------------------------------------------------------
# Features
#------------------------------------------------------------------------------

def test_features_empty():
    _test_empty(FeatureGridViewModel)


def test_features_full():
    _test_view_model(FeatureGridViewModel, marker_size=8, n_spikes_max=20)


def test_features_lasso():
    vm = _test_view_model(FeatureGridViewModel,
                          marker_size=8,
                          stop=False,
                          )
    show_test_run(vm.view, _N_FRAMES)
    box = (1, 2)
    vm.view.lasso.box = box
    x, y = 0., 1.
    vm.view.lasso.add((x, x))
    vm.view.lasso.add((y, x))
    vm.view.lasso.add((y, y))
    vm.view.lasso.add((x, y))
    show_test_run(vm.view, _N_FRAMES)
    # Find spikes in lasso.
    spikes = vm.spikes_in_lasso()
    # Change their clusters.
    vm.model.spike_clusters[spikes] = 3
    sc = vm.model.spike_clusters
    vm.view.visual.spike_clusters = sc[vm.view.visual.spike_ids]
    show_test_run(vm.view, _N_FRAMES)
    show_test_stop(vm.view)


#------------------------------------------------------------------------------
# Correlograms
#------------------------------------------------------------------------------

def test_ccg_empty():
    _test_empty(CorrelogramViewModel,
                binsize=20,
                winsize_bins=51,
                n_excerpts=100,
                excerpt_size=100,
                )


def test_ccg_simple():
    _test_view_model(CorrelogramViewModel,
                     binsize=10,
                     winsize_bins=61,
                     n_excerpts=80,
                     excerpt_size=120,
                     )


def test_ccg_full():
    vm = _test_view_model(CorrelogramViewModel,
                          binsize=20,
                          winsize_bins=51,
                          n_excerpts=100,
                          excerpt_size=100,
                          stop=False,
                          )
    show_test_run(vm.view, _N_FRAMES)
    vm.change_bins(half_width=100., bin=1.)
    show_test_run(vm.view, _N_FRAMES)
    show_test_stop(vm.view)


#------------------------------------------------------------------------------
# Traces
#------------------------------------------------------------------------------

def test_traces_empty():
    _test_empty(TraceViewModel)


def test_traces_simple():
    _test_view_model(TraceViewModel)


def test_traces_full():
    vm = _test_view_model(TraceViewModel, stop=False)
    vm.move_right()
    show_test_run(vm.view, _N_FRAMES)
    vm.move_left()
    show_test_run(vm.view, _N_FRAMES)

    show_test_stop(vm.view)
