# -*- coding: utf-8 -*-

"""Test GUI component."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture
import numpy as np
from numpy.testing import assert_array_equal as ae

from ..gui_component import ManualClustering
from phy.gui import GUI


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def manual_clustering(gui, cluster_ids, cluster_groups):
    spike_clusters = np.array(cluster_ids)

    mc = ManualClustering(spike_clusters,
                          cluster_groups=cluster_groups,
                          shortcuts={'undo': 'ctrl+z'},
                          )
    _s = []

    mc.attach(gui)

    # Connect to the `select` event.
    @mc.gui.connect_
    def on_select(cluster_ids, spike_ids):
        _s.append((cluster_ids, spike_ids))

    def assert_selection(*cluster_ids):  # pragma: no cover
        if not _s:
            return
        assert _s[-1][0] == list(cluster_ids)

    yield mc, assert_selection


@yield_fixture
def gui(qapp):
    gui = GUI(position=(200, 100), size=(500, 500))
    yield gui
    gui.close()


#------------------------------------------------------------------------------
# Test GUI components
#------------------------------------------------------------------------------

def test_manual_clustering_edge_cases(manual_clustering):
    mc, assert_selection = manual_clustering

    # Empty selection at first.
    assert_selection()
    ae(mc.clustering.cluster_ids, [0, 1, 2, 10, 11, 20, 30])

    mc.select([0])
    assert_selection(0)

    mc.undo()
    mc.redo()

    # Merge.
    mc.merge()
    assert_selection(0)

    mc.merge([])
    assert_selection(0)

    mc.merge([10])
    assert_selection(0)

    # Split.
    mc.split([])
    assert_selection(0)

    # Move.
    mc.move([], 'ignored')

    mc.save()


def test_manual_clustering_merge(manual_clustering):
    mc, assert_selection = manual_clustering

    mc.actions.select(30, 20)  # NOTE: we pass multiple ints instead of a list
    mc.actions.merge()
    assert_selection(31, 2)

    mc.actions.undo()
    assert_selection(30, 20)

    mc.actions.redo()
    assert_selection(31, 2)


def test_manual_clustering_split(manual_clustering):
    mc, assert_selection = manual_clustering

    mc.actions.select([1, 2])
    mc.actions.split([1, 2])
    assert_selection(31, 20)

    mc.actions.undo()
    assert_selection(1, 2)

    mc.actions.redo()
    assert_selection(31, 20)


def test_manual_clustering_split_2(gui):
    spike_clusters = np.array([0, 0, 1])

    mc = ManualClustering(spike_clusters)
    mc.attach(gui)

    mc.actions.split([0])
    # assert mc.wizard.selection == [2, 1]


def test_manual_clustering_move(manual_clustering, quality, similarity):
    mc, assert_selection = manual_clustering

    mc.actions.select([30])
    assert_selection(30)

    # mc.wizard.set_quality_function(quality)
    # mc.wizard.set_similarity_function(similarity)

    # mc.actions.next_by_quality()
    # assert_selection(20)

    mc.actions.move([20], 'noise')
    assert_selection(2)

    mc.actions.undo()
    assert_selection(20)

    mc.actions.redo()
    assert_selection(2)


def test_manual_clustering_show(qtbot, gui):
    spike_clusters = np.array([0, 0, 1, 2, 0, 1])

    def sf(c, d):
        return float(c + d)

    mc = ManualClustering(spike_clusters, similarity_func=sf)
    mc.attach(gui)
    gui.show()
    qtbot.stop()
