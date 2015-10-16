# -*- coding: utf-8 -*-

"""Test GUI plugin."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture
import numpy as np
from numpy.testing import assert_array_equal as ae

from ..clustering import Clustering
from .._utils import create_cluster_meta
from ..gui_plugin import (_wizard_group,
                          _attach_wizard,
                          _attach_wizard_to_clustering,
                          _attach_wizard_to_cluster_meta,
                          )
from phy.gui.tests.conftest import gui  # noqa


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture  # noqa
def manual_clustering(gui, cluster_ids, cluster_groups):
    spike_clusters = np.array(cluster_ids)

    mc = gui.attach('ManualClustering',
                    spike_clusters=spike_clusters,
                    cluster_groups=cluster_groups,
                    shortcuts={'undo': 'ctrl+z'},
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

def test_wizard_group():
    assert _wizard_group('noise') == 'ignored'
    assert _wizard_group('mua') == 'ignored'
    assert _wizard_group('good') == 'good'
    assert _wizard_group('unknown') is None
    assert _wizard_group(None) is None


def test_attach_wizard_to_clustering_merge(wizard, cluster_ids):
    clustering = Clustering(np.array(cluster_ids))
    _attach_wizard_to_clustering(wizard, clustering)

    assert wizard.selection == []

    wizard.select([30, 20, 10])
    assert wizard.selection == [30, 20, 10]

    clustering.merge([30, 20])
    # Select the merged cluster along with its most similar one (=pin merged).
    assert wizard.selection == [31, 2]

    # Undo: the previous selection reappears.
    clustering.undo()
    assert wizard.selection == [30, 20, 10]

    # Redo.
    clustering.redo()
    assert wizard.selection == [31, 2]


def test_attach_wizard_to_clustering_split(wizard, cluster_ids):
    clustering = Clustering(np.array(cluster_ids))
    _attach_wizard_to_clustering(wizard, clustering)

    wizard.select([30, 20, 10])
    assert wizard.selection == [30, 20, 10]

    clustering.split([5, 3])
    assert wizard.selection == [31, 30]

    # Undo: the previous selection reappears.
    clustering.undo()
    assert wizard.selection == [30, 20, 10]

    # Redo.
    clustering.redo()
    assert wizard.selection == [31, 30]


def test_attach_wizard_to_cluster_meta(wizard, cluster_groups):
    cluster_meta = create_cluster_meta(cluster_groups)
    _attach_wizard_to_cluster_meta(wizard, cluster_meta)

    wizard.select([30])

    wizard.select([20])
    assert wizard.selection == [20]

    cluster_meta.set('group', [20], 'noise')
    assert cluster_meta.get('group', 20) == 'noise'
    assert wizard.selection == [2]

    cluster_meta.set('group', [2], 'good')
    assert wizard.selection == [11]

    # Restart.
    wizard.restart()
    assert wizard.selection == [30]

    # 30, 20, 11, 10, 2, 1, 0
    #  N,  i,  g,  i, g, g, i
    assert wizard.next_by_quality() == [11]
    assert wizard.next_by_quality() == [2]
    assert wizard.next_by_quality() == [1]
    assert wizard.next_by_quality() == [20]
    assert wizard.next_by_quality() == [10]
    assert wizard.next_by_quality() == [0]


def test_attach_wizard_to_cluster_meta_undo(wizard, cluster_groups):
    cluster_meta = create_cluster_meta(cluster_groups)
    _attach_wizard_to_cluster_meta(wizard, cluster_meta)

    wizard.select([20])

    cluster_meta.set('group', [20], 'noise')
    assert wizard.selection == [2]

    wizard.next_by_quality()
    assert wizard.selection == [11]

    cluster_meta.undo()
    assert wizard.selection == [20]

    cluster_meta.redo()
    assert wizard.selection == [2]


def test_attach_wizard_1(wizard, cluster_ids, cluster_groups):
    clustering = Clustering(np.array(cluster_ids))
    cluster_meta = create_cluster_meta(cluster_groups)
    _attach_wizard(wizard, clustering, cluster_meta)

    wizard.restart()
    assert wizard.selection == [30]

    wizard.pin()
    assert wizard.selection == [30, 20]

    clustering.merge(wizard.selection)
    assert wizard.selection == [31, 2]
    assert cluster_meta.get('group', 31) is None

    wizard.next_by_quality()
    assert wizard.selection == [31, 11]

    clustering.undo()
    assert wizard.selection == [30, 20]


def test_attach_wizard_2(wizard, cluster_ids, cluster_groups):
    clustering = Clustering(np.array(cluster_ids))
    cluster_meta = create_cluster_meta(cluster_groups)
    _attach_wizard(wizard, clustering, cluster_meta)

    wizard.select([30, 20])
    assert wizard.selection == [30, 20]

    clustering.split([1])
    assert wizard.selection == [31, 30]
    assert cluster_meta.get('group', 31) is None

    wizard.next_by_quality()
    assert wizard.selection == [31, 20]

    clustering.undo()
    assert wizard.selection == [30, 20]


def test_attach_wizard_3(wizard, cluster_ids, cluster_groups):
    clustering = Clustering(np.array(cluster_ids))
    cluster_meta = create_cluster_meta(cluster_groups)
    _attach_wizard(wizard, clustering, cluster_meta)

    wizard.select([30, 20])
    assert wizard.selection == [30, 20]

    cluster_meta.set('group', 30, 'noise')
    assert wizard.selection == [20]


#------------------------------------------------------------------------------
# Test GUI plugins
#------------------------------------------------------------------------------

def test_wizard_start_1(manual_clustering):
    mc, assert_selection = manual_clustering

    # Check that the wizard_start event is fired.
    _check = []

    @mc.gui.connect_
    def on_wizard_start():
        _check.append('wizard')

    mc.wizard.restart()
    assert _check == ['wizard']


def test_wizard_start_2(manual_clustering):
    mc, assert_selection = manual_clustering

    # Check that the wizard_start event is fired.
    _check = []

    @mc.gui.connect_
    def on_wizard_start():
        _check.append('wizard')

    mc.wizard.select([1])
    assert _check == ['wizard']


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

    mc.actions.select([30, 20])
    mc.actions.merge()
    assert_selection(31, 2)


def test_manual_clustering_split(manual_clustering):
    mc, assert_selection = manual_clustering

    mc.actions.select([1, 2])
    mc.actions.split([1, 2])
    assert_selection(31, 20)


def test_manual_clustering_move(manual_clustering, quality, similarity):
    mc, assert_selection = manual_clustering

    mc.actions.select([30])
    assert_selection(30)

    mc.wizard.set_quality_function(quality)
    mc.wizard.set_similarity_function(similarity)

    mc.actions.next_by_quality()
    assert_selection(20)

    mc.actions.move([20], 'noise')
    assert_selection(2)
