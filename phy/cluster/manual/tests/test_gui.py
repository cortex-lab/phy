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


def test_gui_multiple_clusterings():

    n_clusters = 5
    n_spikes = 100
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

        assert gui.model.n_spikes == n_spikes
        assert gui.model.n_clusters == n_clusters
        assert len(gui.model.cluster_ids) == n_clusters
        assert gui.clustering.n_clusters == n_clusters
        assert gui.model.cluster_metadata.group(1) == 1

        # Change clustering.
        with raises(ValueError):
            gui.change_clustering('automat')
        gui.change_clustering('automatic')

        assert gui.model.n_spikes == n_spikes
        assert gui.model.n_clusters == n_clusters * 2
        assert len(gui.model.cluster_ids) == n_clusters * 2
        assert gui.clustering.n_clusters == n_clusters * 2
        assert gui.model.cluster_metadata.group(2) == 2

        # Merge the clusters and save, for the current clustering.
        gui.clustering.merge(gui.clustering.cluster_ids)
        gui.save()
        gui.close()

        # Re-open the gui.
        gui = _start_manual_clustering(kwik_path=kwik_path,
                                           tempdir=tempdir)
        # The default clustering is the main one: nothing should have
        # changed here.
        assert gui.model.n_clusters == n_clusters
        gui.change_clustering('automatic')
        assert gui.model.n_spikes == n_spikes
        assert gui.model.n_clusters == 1
        assert gui.model.cluster_ids == n_clusters * 2


@mark.long
def test_gui_mock():
    with TemporaryDirectory() as tempdir:
        gui = _start_manual_clustering(model=MockModel(),
                                           tempdir=tempdir)
        for name in ('waveforms', 'features', 'correlograms', 'traces'):
            vm = _show_view(gui, name, [], stop=False)
            vm.select([0])
            show_test_run(vm.view, _N_FRAMES)
            vm.select([0, 1])
            show_test_run(vm.view, _N_FRAMES)
            show_test_stop(vm.view)


def test_gui_kwik():
    n_clusters = 5
    n_spikes = 50
    n_channels = 28
    n_fets = 2
    n_samples_traces = 3000

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

        # Check backup.
        assert op.exists(op.join(tempdir, kwik_path + '.bak'))

        cs = gui.cluster_store

        nc = n_channels - 2

        # Check the stored items.
        for cluster in range(n_clusters):
            n_spikes = len(gui.clustering.spikes_per_cluster[cluster])
            n_unmasked_channels = cs.n_unmasked_channels(cluster)

            assert cs.features(cluster).shape == (n_spikes, nc, n_fets)
            assert cs.masks(cluster).shape == (n_spikes, nc)
            assert cs.mean_masks(cluster).shape == (nc,)
            assert n_unmasked_channels <= nc
            assert cs.mean_probe_position(cluster).shape == (2,)
            assert cs.main_channels(cluster).shape == (n_unmasked_channels,)

        # _show_view(gui, 'waveforms', [0])
        # _show_view(gui, 'features', [0])

        gui.close()


def test_gui_wizard():

    n_clusters = 5
    n_spikes = 100
    n_channels = 28
    n_fets = 2
    n_samples_traces = 3000

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

        clusters = np.arange(n_clusters)
        best_clusters = gui.wizard.best_clusters()

        # assert gui.wizard.best_clusters(1)[0] == best_clusters[0]
        ae(np.unique(best_clusters), clusters)
        assert len(gui.wizard.most_similar_clusters()) == n_clusters - 1

        assert len(gui.wizard.most_similar_clusters(0, n_max=3)) == 3

        gui.merge([0, 1, 2])
        assert np.all(np.in1d(gui.wizard.best_clusters(), np.arange(3, 6)))
        assert list(gui.wizard.most_similar_clusters(5)) in ([3, 4],
                                                                 [4, 3])

        # Move a cluster to noise.
        gui.move([5], 0)
        best = gui.wizard.best_clusters(1)[0]
        if best is not None:
            assert best in (3, 4)
            # The most similar cluster is 3 if best=4 and conversely.
            assert gui.wizard.most_similar_clusters(best)[0] == 7 - best


def test_gui_statistics():
    """Test registration of new statistic."""
    n_clusters = 5
    n_spikes = 100
    n_channels = 28
    n_fets = 2
    n_samples_traces = 3000

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

        @gui.register_statistic
        def n_spikes_2(cluster):
            return gui.clustering.cluster_counts.get(cluster, 0) ** 2

        store = gui.cluster_store
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
