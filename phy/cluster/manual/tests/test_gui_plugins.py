# -*- coding: utf-8 -*-

"""Test GUI plugins."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture
import numpy as np
from numpy.testing import assert_array_equal as ae

from ..clustering import Clustering
from .._utils import create_cluster_meta
from ..gui_plugins import (_attach_wizard,
                           _attach_wizard_to_clustering,
                           _attach_wizard_to_cluster_meta,
                           )
from phy.gui.tests.test_gui import gui  # noqa


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture  # noqa
def manual_clustering(qtbot, gui, cluster_ids, cluster_groups):
    spike_clusters = np.array(cluster_ids)

    mc = gui.attach('ManualClustering',
                    spike_clusters=spike_clusters,
                    cluster_groups=cluster_groups,
                    )

    _s = []

    # Connect to the `select` event.
    @mc.gui.connect_
    def on_select(cluster_ids, spike_ids):
        _s.append((cluster_ids, spike_ids))

    def assert_selection(*cluster_ids):  # pragma: no cover
        if not _s:
            return
        assert _s[-1][0] == list(cluster_ids)
        if len(cluster_ids) >= 1:
            assert mc.wizard.best == cluster_ids[0]
        elif len(cluster_ids) >= 2:
            assert mc.wizard.match == cluster_ids[2]

    yield mc, assert_selection


#------------------------------------------------------------------------------
# Test wizard attach
#------------------------------------------------------------------------------

def test_attach_wizard_to_clustering_merge(wizard, cluster_ids):
    clustering = Clustering(np.array(cluster_ids))
    _attach_wizard_to_clustering(wizard, clustering)

    assert wizard.selection == []

    wizard.select([30, 20, 10])
    assert wizard.selection == [30, 20, 10]

    clustering.merge([30, 20])
    # Select the merged cluster along with its most similar one (=pin merged).
    assert wizard.selection == [31, 10]

    # Undo: the previous selection reappears.
    clustering.undo()
    assert wizard.selection == [30, 20, 10]

    # Redo.
    clustering.redo()
    assert wizard.selection == [31, 10]


def test_attach_wizard_to_clustering_split(wizard, cluster_ids):
    clustering = Clustering(np.array(cluster_ids))
    _attach_wizard_to_clustering(wizard, clustering)

    wizard.select([30, 20, 10])
    assert wizard.selection == [30, 20, 10]

    clustering.split([5, 3])
    assert wizard.selection == [31, 20]

    # Undo: the previous selection reappears.
    clustering.undo()
    assert wizard.selection == [30, 20, 10]

    # Redo.
    clustering.redo()
    assert wizard.selection == [31, 20]


def test_attach_wizard_to_cluster_meta(wizard, cluster_groups):
    cluster_meta = create_cluster_meta(cluster_groups)
    _attach_wizard_to_cluster_meta(wizard, cluster_meta)

    wizard.select([30])

    wizard.select([20])
    assert wizard.selection == [20]

    cluster_meta.set('group', [20], 'noise')
    # assert wizard.selection == [10]


def test_attach_wizard(wizard, cluster_ids, cluster_groups):
    clustering = Clustering(np.array(cluster_ids))
    cluster_meta = create_cluster_meta(cluster_groups)
    _attach_wizard(wizard, clustering, cluster_meta)


#------------------------------------------------------------------------------
# Test GUI plugins
#------------------------------------------------------------------------------

def test_manual_clustering_edge_cases(manual_clustering):
    mc, assert_selection = manual_clustering

    # Empty selection at first.
    assert_selection()
    ae(mc.clustering.cluster_ids, [0, 1, 2, 10, 20, 30])

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

    mc.actions.select([30, 20])
    mc.actions.merge()
    # assert_selection(31, 10)
