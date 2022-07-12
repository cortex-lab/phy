# -*- coding: utf-8 -*-

"""Test supervisor."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture

from .. import supervisor as _supervisor
from ..supervisor import (
    Supervisor, ActionCreator, TableController)
from phy.gui import GUI
from phy.gui.qt import qInstallMessageHandler


def handler(msg_type, msg_log_context, msg_string):
    pass


qInstallMessageHandler(handler)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def gui(tempdir, qtbot):
    # NOTE: mock patch show box exec_
    _supervisor._show_box = lambda _: _

    gui = GUI(position=(200, 100), size=(800, 600), config_dir=tempdir)
    gui.set_default_actions()
    gui.show()
    qtbot.waitForWindowShown(gui)
    yield gui
    qtbot.wait(5)
    gui.close()
    del gui
    qtbot.wait(5)


@fixture
def controller(gui, cluster_ids, cluster_groups, cluster_labels, cluster_metrics, similarity):
    c = TableController(
        cluster_ids=cluster_ids,
        cluster_groups=cluster_groups,
        cluster_labels=cluster_labels,
        cluster_metrics=cluster_metrics,
        similarity=similarity,
    )
    c.attach(gui)
    return c


@fixture
def supervisor(gui, spike_clusters, cluster_groups, cluster_labels, cluster_metrics, similarity):
    s = Supervisor(
        spike_clusters=spike_clusters,
        cluster_groups=cluster_groups,
        cluster_metrics=cluster_metrics,
        cluster_labels=cluster_labels,
        similarity=similarity,
    )
    s.attach(gui)
    return s


#------------------------------------------------------------------------------
# Action creator tests
#------------------------------------------------------------------------------

def test_action_creator_1(qtbot, gui):
    ac = ActionCreator()
    ac.attach(gui)
    gui.show()
    # qtbot.stop()


#------------------------------------------------------------------------------
# Table controller tests
#------------------------------------------------------------------------------

def test_table_controller_1(qtbot, gui, controller):
    c = controller
    assert len(c.cluster_info()) == 7

    assert c.shown_cluster_ids == [30, 20, 11, 10, 2, 1, 0]
    assert c.similarity_view.shown_ids() == []

    c.select_clusters([30, 11])
    assert c.selected_clusters == [30, 11]
    assert c.selected_similar == []
    assert c.similarity_view.shown_ids() == [20, 10, 2, 1, 0]

    c.select_similar([10, 1])
    assert c.selected_similar == [10, 1]

    # qtbot.stop()


def test_table_controller_2(qtbot, gui, controller):
    c = controller

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N,  N

    # Add a cluster.
    c.add_cluster(100, n_spikes=3, group='mygroup', test_label='mylabel')

    # Select a cluster.
    assert 100 in c.shown_cluster_ids
    c.select_clusters([100])
    assert c.selected_clusters == [100]

    # Change a cluster.
    c.change_cluster(11, group='noise', n_spikes=60)
    assert 11 in c.shown_cluster_ids

    # Remove a cluster.
    assert 20 in c.shown_cluster_ids
    c.remove_cluster(20)
    assert 20 not in c.shown_cluster_ids

    # qtbot.stop()


def test_table_controller_3(qtbot, gui, controller):
    c = controller

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N,  N

    c.add_column('new_column')
    assert 'new_column' in c.columns

    c.select_clusters([30])

    # qtbot.stop()


def test_table_controller_4(qtbot, gui, controller):
    c = controller

    # [0, 1, 2, 10, 11, 20, 30]
    #  i, g, N,  i,  g,  N,  N

    c.remove_column('test_label')
    assert 'test_label' not in c.columns

    # qtbot.stop()


#------------------------------------------------------------------------------
# Supervisor tests
#------------------------------------------------------------------------------

def _a(tc, c, s):
    assert tc.selected_clusters == c
    assert tc.selected_similar == s


def test_supervisor_1(qtbot, gui, supervisor):
    s = supervisor
    tc = supervisor.table_controller

    s.on_next_best()
    _a(tc, [30], [])

    s.on_next_best()
    _a(tc, [20], [])

    s.on_next()
    _a(tc, [20], [30])

    s.on_prev()
    _a(tc, [20], [30])

    s.on_prev_best()
    _a(tc, [30], [20])


def test_supervisor_2(qtbot, gui, supervisor):
    s = supervisor
    tc = supervisor.table_controller

    s.on_next()
    _a(tc, [30], [20])

    s.on_next()
    _a(tc, [30], [11])

    s.on_undo()
    _a(tc, [30], [11])

    s.on_prev()
    _a(tc, [30], [20])

    s.on_merge()
    _a(tc, [31], [])

    s.on_next()
    _a(tc, [31], [11])

    s.on_undo()
    _a(tc, [30], [20])

    s.on_redo()
    _a(tc, [31], [])


def test_supervisor_3(qtbot, gui, supervisor):
    s = supervisor
    tc = supervisor.table_controller

    s.on_next()
    _a(tc, [30], [20])

    s.on_move('mua', 'similar')
    _a(tc, [30], [11])

    s.on_move('mua', 'best')
    _a(tc, [20], [11])
    print(tc.selected_clusters, tc.selected_similar)

    # qtbot.stop()
