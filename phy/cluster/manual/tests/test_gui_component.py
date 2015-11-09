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
def manual_clustering(qtbot, gui, cluster_ids, cluster_groups,
                      quality, similarity):
    spike_clusters = np.array(cluster_ids)

    mc = ManualClustering(spike_clusters,
                          cluster_groups=cluster_groups,
                          shortcuts={'undo': 'ctrl+z'},
                          quality_func=quality,
                          similarity_func=similarity,
                          )
    mc.attach(gui)
    qtbot.waitForWindowShown(mc.cluster_view)
    yield mc


@yield_fixture
def gui(qapp):
    gui = GUI(position=(200, 100), size=(500, 500))
    yield gui
    gui.close()


#------------------------------------------------------------------------------
# Test GUI components
#------------------------------------------------------------------------------

def test_manual_clustering_edge_cases(manual_clustering):
    mc = manual_clustering

    # Empty selection at first.
    ae(mc.clustering.cluster_ids, [0, 1, 2, 10, 11, 20, 30])

    mc.select([0])
    assert mc.selected == [0]

    mc.undo()
    mc.redo()

    # Merge.
    mc.merge()
    assert mc.selected == [0]

    mc.merge([])
    assert mc.selected == [0]

    mc.merge([10])
    assert mc.selected == [0]

    # Split.
    mc.split([])
    assert mc.selected == [0]

    # Move.
    mc.move([], 'ignored')

    mc.save()


def test_manual_clustering_merge(manual_clustering):
    mc = manual_clustering

    mc.select(30, 20)  # NOTE: we pass multiple ints instead of a list
    mc.merge()
    assert mc.selected == [31, 2]

    mc.undo()
    assert mc.selected == [30, 20]

    mc.redo()
    assert mc.selected == [31, 2]


def test_manual_clustering_split(manual_clustering):
    mc = manual_clustering

    mc.select([1, 2])
    mc.split([1, 2])
    assert mc.selected == [31, 20]

    mc.undo()
    assert mc.selected == [1, 2]

    mc.redo()
    assert mc.selected == [31, 20]


def test_manual_clustering_split_2(gui):
    spike_clusters = np.array([0, 0, 1])

    mc = ManualClustering(spike_clusters)
    mc.attach(gui)

    mc.split([0])
    # assert mc.wizard.selection == [2, 1]


def test_manual_clustering_move(manual_clustering, quality, similarity):
    mc = manual_clustering

    mc.select([30])
    assert mc.selected == [30]

    mc.cluster_view.next()
    assert mc.selected == [20]

    mc.move([20], 'noise')
    assert mc.selected == [2]

    mc.undo()
    assert mc.selected == [20]

    mc.redo()
    assert mc.selected == [2]


# def test_manual_clustering_show(qtbot, gui):
#     mc, assert_selection = manual_clustering
