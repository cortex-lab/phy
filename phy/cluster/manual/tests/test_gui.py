# -*- coding: utf-8 -*-

"""Tests of manual clustering GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_allclose as ac
from numpy.testing import assert_array_equal as ae
from pytest import raises, mark

from ..gui import ClusterManualGUI
from ....utils import _spikes_in_clusters
from ....utils.testing import (show_test_start, show_test_run,
                               show_test_stop,
                               )
from ....gui.qt import qt_app, _close_qt_after, wrap_qt
from ....utils.tempdir import TemporaryDirectory
from ....utils.logging import set_level
from ....io.mock import MockModel
from ....io.kwik.mock import create_mock_kwik
from ....io.kwik.store_items import create_store


#------------------------------------------------------------------------------
# Kwik tests
#------------------------------------------------------------------------------

_N_FRAMES = 2


def setup():
    set_level('info')


def _start_manual_clustering():
    model = MockModel()
    spc = model.spikes_per_cluster
    store = create_store(model, spikes_per_cluster=spc)
    gui = ClusterManualGUI(model=model, store=store, config=[])
    return gui


def _show_view(gui, name, cluster_ids=None, stop=True):
    vm = gui.show_view(name, cluster_ids, show=False, scale_factor=1)
    show_test_start(vm.view)
    show_test_run(vm.view, _N_FRAMES)
    if stop:
        show_test_stop(vm.view)
    return vm


@wrap_qt
def test_gui_clustering():

    gui = _start_manual_clustering()
    gui.show()
    yield

    cs = gui.store
    spike_clusters = gui.model.spike_clusters.copy()

    f = gui.model.features
    m = gui.model.masks

    def _check_arrays(cluster, clusters_for_sc=None, spikes=None):
        """Check the features and masks in the cluster store
        of a given custer."""
        if spikes is None:
            if clusters_for_sc is None:
                clusters_for_sc = [cluster]
            spikes = _spikes_in_clusters(spike_clusters, clusters_for_sc)
        shape = (len(spikes),
                 len(gui.model.channel_order),
                 gui.model.n_features_per_channel)
        ac(cs.features(cluster), f[spikes, :].reshape(shape))
        ac(cs.masks(cluster), m[spikes])

    _check_arrays(0)
    _check_arrays(2)

    # Merge two clusters.
    clusters = [0, 2]
    up = gui.merge(clusters)
    new = up.added[0]
    _check_arrays(new, clusters)

    # Split some spikes.
    spikes = [2, 3, 5, 7, 11, 13]
    # clusters = np.unique(spike_clusters[spikes])
    up = gui.split(spikes)
    _check_arrays(new + 1, spikes=spikes)

    # Undo.
    gui.undo()
    _check_arrays(new, clusters)

    # Undo.
    gui.undo()
    _check_arrays(0)
    _check_arrays(2)

    # Redo.
    gui.redo()
    _check_arrays(new, clusters)

    # Split some spikes.
    spikes = [5, 7, 11, 13, 17, 19]
    # clusters = np.unique(spike_clusters[spikes])
    gui.split(spikes)
    _check_arrays(new + 1, spikes=spikes)

    # Test merge-undo-different-merge combo.
    spc = gui.clustering.spikes_per_cluster.copy()
    clusters = gui.cluster_ids[:3]
    up = gui.merge(clusters)
    _check_arrays(up.added[0], spikes=up.spike_ids)
    # Undo.
    gui.undo()
    for cluster in clusters:
        _check_arrays(cluster, spikes=spc[cluster])
    # Another merge.
    clusters = gui.cluster_ids[1:5]
    up = gui.merge(clusters)
    _check_arrays(up.added[0], spikes=up.spike_ids)

    # Move a cluster to a group.
    cluster = gui.cluster_ids[0]
    gui.move([cluster], 2)
    assert len(gui.store.mean_probe_position(cluster)) == 2

    spike_clusters_new = gui.model.spike_clusters.copy()
    # Check that the spike clusters have changed.
    assert not np.all(spike_clusters_new == spike_clusters)
    ac(gui.model.spike_clusters, gui.clustering.spike_clusters)

    gui.close()


@wrap_qt
def test_gui_wizard():
    gui = _start_manual_clustering()
    n = gui.n_clusters
    gui.show()
    yield

    clusters = np.arange(gui.n_clusters)
    best_clusters = gui.wizard.best_clusters()

    # assert gui.wizard.best_clusters(1)[0] == best_clusters[0]
    ae(np.unique(best_clusters), clusters)
    assert len(gui.wizard.most_similar_clusters()) == n - 1

    assert len(gui.wizard.most_similar_clusters(0, n_max=3)) == 3

    clusters = gui.cluster_ids[:2]
    up = gui.merge(clusters)
    new = up.added[0]
    assert np.all(np.in1d(gui.wizard.best_clusters(),
                  np.arange(clusters[-1] + 1, new + 1)))
    assert np.all(np.in1d(gui.wizard.most_similar_clusters(new),
                  np.arange(clusters[-1] + 1, new)))

    gui.close()


@wrap_qt
def test_gui_statistics():
    """Test registration of new statistic."""

    gui = _start_manual_clustering()
    gui.show()
    yield

    @gui.register_statistic
    def n_spikes_2(cluster):
        return gui.clustering.cluster_counts.get(cluster, 0) ** 2

    store = gui.store
    stats = store.items['statistics']

    def _check():
        for clu in gui.cluster_ids:
            assert (store.n_spikes_2(clu) ==
                    store.features(clu).shape[0] ** 2)

    assert 'n_spikes_2' in stats.fields
    _check()

    # Merge the clusters and check that the statistics has been
    # recomputed for the new cluster.
    clusters = gui.cluster_ids
    gui.merge(clusters)
    _check()
    assert gui.cluster_ids == [max(clusters) + 1]

    gui.undo()
    _check()

    gui.merge(gui.cluster_ids[::2])
    _check()

    gui.close()


@mark.long
def test_gui_history():

    n_clusters = 15
    n_spikes = 300
    n_channels = 28
    n_fets = 2
    n_samples_traces = 10000

    with TemporaryDirectory() as tempdir:

        # Create the test HDF5 file in the temporary directory.
        kwik_path = create_mock_kwik(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)

        gui = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)

        gui.wizard.start()

        spikes = _spikes_in_clusters(gui.model.spike_clusters,
                                     gui.wizard.selection)
        gui.split(spikes[::3])
        gui.undo()
        gui.wizard.next()
        gui.redo()
        gui.undo()

        for _ in range(10):
            gui.merge(gui.wizard.selection)
            gui.wizard.next()
            gui.undo()
            gui.wizard.next()
            gui.redo()
            gui.wizard.next()

            spikes = _spikes_in_clusters(gui.model.spike_clusters,
                                         gui.wizard.selection)
            if len(spikes):
                gui.split(spikes[::10])
                gui.wizard.next()
                gui.undo()
            gui.merge(gui.wizard.selection)
            gui.wizard.next()
            gui.wizard.next()

            gui.wizard.next_best()
            ae(gui.model.spike_clusters, gui.clustering.spike_clusters)


@mark.long
def test_gui_gui():
    n_clusters = 15
    n_spikes = 500
    n_channels = 30
    n_fets = 3
    n_samples_traces = 50000

    with TemporaryDirectory() as tempdir:

        # Create the test HDF5 file in the temporary directory.
        kwik_path = create_mock_kwik(tempdir,
                                     n_clusters=n_clusters,
                                     n_spikes=n_spikes,
                                     n_channels=n_channels,
                                     n_features_per_channel=n_fets,
                                     n_samples_traces=n_samples_traces)

        gui = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)

        with qt_app():
            config = gui.settings['gui_config']
            for name, kwargs in config:
                if name in ('waveforms', 'features', 'traces'):
                    kwargs['scale_factor'] = 1
            gui = gui.gui_creator.add(config=config, show=False)
            _close_qt_after(gui, 0.25)
            gui.show()
