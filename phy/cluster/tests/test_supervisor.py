# -*- coding: utf-8 -*-

"""Test GUI component."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture, fixture
import numpy as np
from numpy.testing import assert_array_equal as ae

from .. import supervisor as _supervisor
from ..supervisor import (Supervisor,
                          )
from phy.io import Context
from phy.gui import GUI


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def gui(tempdir, qtbot):
    # NOTE: mock patch show box exec_
    _supervisor._show_box = lambda _: _

    gui = GUI(position=(200, 100), size=(500, 500), config_dir=tempdir)
    gui.show()
    qtbot.waitForWindowShown(gui)
    yield gui
    qtbot.wait(5)
    gui.close()
    del gui
    qtbot.wait(5)


@fixture
def supervisor(qtbot, gui, cluster_ids, cluster_groups,
               quality, similarity,
               tempdir):
    spike_clusters = np.array(cluster_ids)

    mc = Supervisor(spike_clusters,
                    cluster_groups=cluster_groups,
                    shortcuts={'undo': 'ctrl+z'},
                    quality=quality,
                    similarity=similarity,
                    context=Context(tempdir),
                    )
    mc.attach(gui)
    mc.set_default_sort(quality.__name__)

    return mc


#------------------------------------------------------------------------------
# Test GUI component
#------------------------------------------------------------------------------

def test_supervisor_order(supervisor):
    mc = supervisor
    mc.select([1, 0])
    assert mc.selected == [1, 0]


def test_supervisor_edge_cases(supervisor):
    mc = supervisor

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


def test_supervisor_skip(qtbot, gui, supervisor):
    mc = supervisor

    # yield [0, 1, 2, 10, 11, 20, 30]
    # #      i, g, N,  i,  g,  N, N
    expected = [30, 20, 11, 2, 1]

    for clu in expected:
        mc.cluster_view.next()
        assert mc.selected == [clu]


def test_supervisor_merge(supervisor):
    mc = supervisor

    mc.cluster_view.select([30])
    mc.similarity_view.select([20])
    assert mc.selected == [30, 20]

    mc.merge()
    assert mc.selected == [31, 11]

    mc.undo()
    assert mc.selected == [30, 20]

    mc.redo()
    assert mc.selected == [31, 11]


def test_supervisor_merge_move(supervisor):
    """Check that merge then move selects the next cluster in the original
    cluster view, not the updated cluster view."""
    mc = supervisor

    mc.cluster_view.select([20, 11])

    mc.merge()
    assert mc.selected == [31]

    mc.move('good')
    assert mc.selected == [2]

    mc.cluster_view.select([30])

    mc.move('good')
    assert mc.selected == [2]


def test_supervisor_split_0(supervisor):
    mc = supervisor

    mc.select([1, 2])
    mc.split([1, 2])
    assert mc.selected == [31]

    mc.undo()
    assert mc.selected == [1, 2]

    mc.redo()
    assert mc.selected == [31]


def test_supervisor_split_1(gui, supervisor):
    mc = supervisor
    mc.select([1, 2])

    @gui.connect_
    def on_request_split():
        return mc.clustering.spikes_in_clusters([1, 2])

    mc.split()
    assert mc.selected == [31]


def test_supervisor_split_2(gui, quality, similarity):
    spike_clusters = np.array([0, 0, 1])

    mc = Supervisor(spike_clusters,
                    similarity=similarity,
                    )
    mc.attach(gui)

    mc.add_column(quality, name='quality', default=True)
    mc.set_default_sort('quality', 'desc')

    mc.split([0])
    assert mc.selected == [3, 2]


def test_supervisor_state(tempdir, qtbot, gui, supervisor):
    mc = supervisor
    cv = mc.cluster_view
    cv.sort_by('id')
    gui.close()
    assert cv.state['sort_by'] == ('id', 'asc')
    cv.set_state(cv.state)
    assert cv.state['sort_by'] == ('id', 'asc')


def test_supervisor_label(supervisor):
    mc = supervisor

    mc.select([20])
    mc.label("my_field", 3.14)

    mc.save()

    assert 'my_field' in mc.fields
    assert mc.get_labels('my_field')[20] == 3.14


def test_supervisor_move_1(supervisor):
    mc = supervisor

    mc.select([20])
    assert mc.selected == [20]

    assert not mc.move('', '')

    mc.move('noise')
    assert mc.selected == [11]

    mc.undo()
    assert mc.selected == [20]

    mc.redo()
    assert mc.selected == [11]


def test_supervisor_move_2(supervisor):
    mc = supervisor

    mc.select([20])
    mc.similarity_view.select([10])

    assert mc.selected == [20, 10]

    mc.move('noise', 10)
    assert mc.selected == [20, 2]

    mc.undo()
    assert mc.selected == [20, 10]

    mc.redo()
    assert mc.selected == [20, 2]


#------------------------------------------------------------------------------
# Test shortcuts
#------------------------------------------------------------------------------

def test_supervisor_action_reset(qtbot, supervisor):
    mc = supervisor

    mc.actions.select([10, 11])

    mc.actions.reset()
    assert mc.selected == [30]

    mc.actions.next()
    assert mc.selected == [30, 20]

    mc.actions.next()
    assert mc.selected == [30, 11]

    mc.actions.previous()
    assert mc.selected == [30, 20]


def test_supervisor_action_nav(qtbot, supervisor):
    mc = supervisor

    mc.actions.reset()
    assert mc.selected == [30]

    mc.actions.next_best()
    assert mc.selected == [20]

    mc.actions.previous_best()
    assert mc.selected == [30]


def test_supervisor_action_move_1(qtbot, supervisor):
    mc = supervisor

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


def test_supervisor_action_move_2(supervisor):
    mc = supervisor

    mc.select([30])
    mc.similarity_view.select([20])

    assert mc.selected == [30, 20]
    mc.actions.move_similar_to_noise()

    assert mc.selected == [30, 11]
    mc.actions.move_similar_to_mua()

    assert mc.selected == [30, 2]
    mc.actions.move_similar_to_good()

    assert mc.selected == [30, 1]

    mc.cluster_meta.get('group', 20) == 'noise'
    mc.cluster_meta.get('group', 11) == 'mua'
    mc.cluster_meta.get('group', 2) == 'good'


def test_supervisor_action_move_3(supervisor):
    mc = supervisor

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
