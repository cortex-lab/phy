# -*- coding: utf-8 -*-

"""Test GUI component."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture, fixture
import numpy as np
from numpy.testing import assert_array_equal as ae
from vispy.util import keys

from .. import gui_component
from ..gui_component import (ManualClustering,
                             )
from phy.io.array import _spikes_in_clusters
from phy.gui import GUI
from .conftest import MockController


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def gui(tempdir, qtbot):
    # NOTE: mock patch show box exec_
    gui_component._show_box = lambda _: _

    gui = GUI(position=(200, 100), size=(500, 500), config_dir=tempdir)
    gui.show()
    qtbot.waitForWindowShown(gui)
    yield gui
    qtbot.wait(5)
    gui.close()
    del gui
    qtbot.wait(5)


@fixture
def manual_clustering(qtbot, gui, cluster_ids, cluster_groups,
                      quality, similarity):
    spike_clusters = np.array(cluster_ids)
    spikes_per_cluster = lambda c: [c]

    mc = ManualClustering(spike_clusters,
                          spikes_per_cluster,
                          cluster_groups=cluster_groups,
                          shortcuts={'undo': 'ctrl+z'},
                          quality=quality,
                          similarity=similarity,
                          )
    mc.attach(gui)
    mc.set_default_sort(quality.__name__)

    return mc


#------------------------------------------------------------------------------
# Test GUI component
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
    mc.move('ignored', [])

    mc.save()


def test_manual_clustering_skip(qtbot, gui, manual_clustering):
    mc = manual_clustering

    # yield [0, 1, 2, 10, 11, 20, 30]
    # #      i, g, N,  i,  g,  N, N
    expected = [30, 20, 11, 2, 1]

    for clu in expected:
        mc.cluster_view.next()
        assert mc.selected == [clu]


def test_manual_clustering_merge(manual_clustering):
    mc = manual_clustering

    mc.cluster_view.select([30])
    mc.similarity_view.select([20])
    assert mc.selected == [30, 20]

    mc.merge()
    assert mc.selected == [31, 11]

    mc.undo()
    assert mc.selected == [30, 20]

    mc.redo()
    assert mc.selected == [31, 11]


def test_manual_clustering_merge_move(manual_clustering):
    """Check that merge then move selects the next cluster in the original
    cluster view, not the updated cluster view."""
    mc = manual_clustering

    mc.cluster_view.select([20, 11])

    mc.merge()
    assert mc.selected == [31]

    mc.move('good')
    assert mc.selected == [2]

    mc.cluster_view.select([30])

    mc.move('good')
    assert mc.selected == [2]


def test_manual_clustering_split(manual_clustering):
    mc = manual_clustering

    mc.select([1, 2])
    mc.split([1, 2])
    assert mc.selected == [31]

    mc.undo()
    assert mc.selected == [1, 2]

    mc.redo()
    assert mc.selected == [31]


def test_manual_clustering_split_2(gui, quality, similarity):
    spike_clusters = np.array([0, 0, 1])

    mc = ManualClustering(spike_clusters,
                          lambda c: _spikes_in_clusters(spike_clusters, [c]),
                          similarity=similarity,
                          )
    mc.attach(gui)

    mc.add_column(quality, name='quality', default=True)
    mc.set_default_sort('quality', 'desc')

    mc.split([0])
    assert mc.selected == [3, 2]


def test_manual_clustering_state(tempdir, qtbot, gui, manual_clustering):
    mc = manual_clustering
    cv = mc.cluster_view
    cv.sort_by('id')
    gui.close()
    assert cv.state['sort_by'] == ('id', 'asc')
    cv.set_state(cv.state)
    assert cv.state['sort_by'] == ('id', 'asc')


def test_manual_clustering_split_lasso(tempdir, qtbot):
    controller = MockController(config_dir=tempdir)
    gui = controller.create_gui()
    mc = controller.manual_clustering
    view = gui.list_views('FeatureView', is_visible=False)[0]

    gui.show()

    # Select one cluster.
    mc.select(0)

    # Simulate a lasso.
    ev = view.events
    ev.mouse_press(pos=(210, 1), button=1, modifiers=(keys.CONTROL,))
    ev.mouse_press(pos=(320, 1), button=1, modifiers=(keys.CONTROL,))
    ev.mouse_press(pos=(320, 30), button=1, modifiers=(keys.CONTROL,))
    ev.mouse_press(pos=(210, 30), button=1, modifiers=(keys.CONTROL,))

    ups = []

    @mc.clustering.connect
    def on_cluster(up):
        ups.append(up)

    mc.split()
    up = ups[0]
    assert up.description == 'assign'
    assert up.added == [4, 5]
    assert up.deleted == [0]

    # qtbot.stop()
    gui.close()


def test_manual_clustering_label(manual_clustering):
    mc = manual_clustering

    mc.select([20])
    mc.label("my_field", 3.14)

    mc.save()

    assert 'my_field' in mc.fields
    assert mc.get_labels('my_field')[20] == 3.14


def test_manual_clustering_move_1(manual_clustering):
    mc = manual_clustering

    mc.select([20])
    assert mc.selected == [20]

    mc.move('noise')
    assert mc.selected == [11]

    mc.undo()
    assert mc.selected == [20]

    mc.redo()
    assert mc.selected == [11]


def test_manual_clustering_move_2(manual_clustering):
    mc = manual_clustering

    mc.select([20])
    mc.similarity_view.select([10])

    assert mc.selected == [20, 10]

    mc.move('noise', 10)
    assert mc.selected == [20, 30]

    mc.undo()
    assert mc.selected == [20, 10]

    mc.redo()
    assert mc.selected == [20, 30]


#------------------------------------------------------------------------------
# Test shortcuts
#------------------------------------------------------------------------------

def test_manual_clustering_action_reset(qtbot, manual_clustering):
    mc = manual_clustering

    mc.actions.select([10, 11])

    mc.actions.reset()
    assert mc.selected == [30]

    mc.actions.next()
    assert mc.selected == [30, 20]

    mc.actions.next()
    assert mc.selected == [30, 11]

    mc.actions.previous()
    assert mc.selected == [30, 20]


def test_manual_clustering_action_nav(qtbot, manual_clustering):
    mc = manual_clustering

    mc.actions.reset()
    assert mc.selected == [30]

    mc.actions.next_best()
    assert mc.selected == [20]

    mc.actions.previous_best()
    assert mc.selected == [30]


def test_manual_clustering_action_move_1(qtbot, manual_clustering):
    mc = manual_clustering

    mc.actions.next()

    assert mc.selected == [30]
    mc.actions.move_best_to_noise()

    assert mc.selected == [20]
    mc.actions.move_best_to_mua()

    assert mc.selected == [11]
    mc.actions.move_best_to_good()

    assert mc.selected == [2]

    mc.cluster_meta.get('group', 30) == 'noise'
    mc.cluster_meta.get('group', 20) == 'mua'
    mc.cluster_meta.get('group', 11) == 'good'

    # qtbot.stop()


def test_manual_clustering_action_move_2(manual_clustering):
    mc = manual_clustering

    mc.select([30])
    mc.similarity_view.select([20])

    assert mc.selected == [30, 20]
    mc.actions.move_similar_to_noise()

    assert mc.selected == [30, 11]
    mc.actions.move_similar_to_mua()

    assert mc.selected == [30, 2]
    mc.actions.move_similar_to_good()

    assert mc.selected == [30, 2]

    mc.cluster_meta.get('group', 20) == 'noise'
    mc.cluster_meta.get('group', 11) == 'mua'
    mc.cluster_meta.get('group', 2) == 'good'


def test_manual_clustering_action_move_3(manual_clustering):
    mc = manual_clustering

    mc.select([30])
    mc.similarity_view.select([20])

    assert mc.selected == [30, 20]
    mc.actions.move_all_to_noise()
    mc.next()

    assert mc.selected == [11, 2]
    mc.actions.move_all_to_mua()

    assert mc.selected == [1]
    mc.actions.move_all_to_good()

    assert mc.selected == [1]

    mc.cluster_meta.get('group', 30) == 'noise'
    mc.cluster_meta.get('group', 20) == 'noise'

    mc.cluster_meta.get('group', 11) == 'mua'
    mc.cluster_meta.get('group', 10) == 'mua'

    mc.cluster_meta.get('group', 2) == 'good'
    mc.cluster_meta.get('group', 1) == 'good'
