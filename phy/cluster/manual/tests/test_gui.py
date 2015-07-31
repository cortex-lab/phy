# -*- coding: utf-8 -*-

"""Tests of manual clustering GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark
import numpy as np
from numpy.testing import assert_allclose as ac
from numpy.testing import assert_array_equal as ae

from ..gui import ClusterManualGUI
from ....utils.settings import _load_default_settings
from ....utils.logging import set_level
from ....utils.array import _spikes_in_clusters
from ....io.mock import MockModel
from ....io.kwik.store_items import create_store


# Skip these tests in "make test-quick".
pytestmark = mark.long()


#------------------------------------------------------------------------------
# Kwik tests
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def _start_manual_clustering(config='none', shortcuts=None):
    if config is 'none':
        config = []
    model = MockModel()
    spc = model.spikes_per_cluster
    store = create_store(model, spikes_per_cluster=spc)
    gui = ClusterManualGUI(model=model, store=store,
                           config=config, shortcuts=shortcuts)
    return gui


def test_gui_clustering(qtbot):

    gui = _start_manual_clustering()
    gui.show()
    qtbot.addWidget(gui.main_window)

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


def test_gui_move_wizard(qtbot):
    gui = _start_manual_clustering()
    qtbot.addWidget(gui.main_window)
    gui.show()

    gui.next()
    gui.pin()
    gui.next()
    best = gui.wizard.best
    assert gui.selected_clusters[0] == best
    match = gui.selected_clusters[1]
    gui.move([gui.wizard.match], 'mua')
    assert gui.selected_clusters[0] == best
    assert gui.selected_clusters[1] != match

    gui.close()


def test_gui_wizard(qtbot):
    gui = _start_manual_clustering()
    n = gui.n_clusters
    qtbot.addWidget(gui.main_window)
    gui.show()

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


@mark.long
def test_gui_history(qtbot):

    gui = _start_manual_clustering()
    qtbot.addWidget(gui.main_window)
    gui.show()

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

    gui.close()


@mark.long
def test_gui_gui(qtbot):
    settings = _load_default_settings()
    config = settings['cluster_manual_config']
    shortcuts = settings['cluster_manual_shortcuts']

    gui = _start_manual_clustering(config=config,
                                   shortcuts=shortcuts,
                                   )
    qtbot.addWidget(gui.main_window)
    gui.show()
    gui.close()
