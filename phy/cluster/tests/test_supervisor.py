# -*- coding: utf-8 -*-

"""Test GUI component."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

#from contextlib import contextmanager

from pytest import yield_fixture, fixture
import numpy as np
from numpy.testing import assert_array_equal as ae

from .. import supervisor as _supervisor
from ..supervisor import (Supervisor,
                          TaskLogger,
                          ClusterView,
                          SimilarityView,
                          ActionCreator,
                          )
from phy.io import Context
from phy.gui import GUI
from phy.gui.widgets import Barrier
from phy.gui.qt import qInstallMessageHandler
#from phy.gui.tests.test_qt import _block
from phy.gui.tests.test_widgets import _assert, _wait_until_table_ready
from phy.utils import connect, Bunch


def handler(msg_type, msg_log_context, msg_string):
    pass


qInstallMessageHandler(handler)


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
def supervisor(qtbot, gui, cluster_ids, cluster_groups, cluster_labels,
               quality, similarity, tempdir):
    spike_clusters = np.repeat(cluster_ids, 2)

    mc = Supervisor(spike_clusters,
                    cluster_groups=cluster_groups,
                    cluster_labels=cluster_labels,
                    quality=quality,
                    similarity=similarity,
                    context=Context(tempdir),
                    )
    mc.attach(gui)
    b = Barrier()
    connect(b('cluster_view'), event='ready', sender=mc.cluster_view)
    connect(b('similarity_view'), event='ready', sender=mc.similarity_view)
    b.wait()
    return mc


#------------------------------------------------------------------------------
# Test tasks
#------------------------------------------------------------------------------

@fixture
def tl():
    class MockClusterView(object):
        _selected = [0]

        def select(self, cl, callback=None):
            self._selected = cl
            callback((cl, cl[-1] + 1))

        def next(self, callback=None):
            callback(([self._selected[-1] + 1], self._selected[-1] + 2))

        def previous(self, callback=None):  # pragma: no cover
            callback(([self._selected[-1] - 1], self._selected[-1]))

    class MockSimilarityView(MockClusterView):
        pass

    class MockSupervisor(object):
        def merge(self, cluster_ids, to, callback=None):
            callback(Bunch(deleted=cluster_ids, added=[to]))

        def split(self, old_cluster_ids, new_cluster_ids, callback=None):
            callback(Bunch(deleted=old_cluster_ids, added=new_cluster_ids))

        def move(self, which, group, callback=None):
            callback(Bunch(metadata_changed=which, metadata_value=group))

        def undo(self, callback=None):
            callback(Bunch())

        def redo(self, callback=None):
            callback(Bunch())

    out = TaskLogger(MockClusterView(), MockSimilarityView(), MockSupervisor())

    return out


def test_task_1(tl):
    assert tl.last_state(None) is None


def test_task_2(tl):
    tl.enqueue(tl.cluster_view, 'select', [0])
    tl.process()
    assert tl.last_state() == ([0], 1, None, None)


def test_task_3(tl):
    tl.enqueue(tl.cluster_view, 'select', [0])
    tl.enqueue(tl.similarity_view, 'select', [100])
    tl.process()
    assert tl.last_state() == ([0], 1, [100], 101)


def test_task_merge(tl):
    tl.enqueue(tl.cluster_view, 'select', [0])
    tl.enqueue(tl.similarity_view, 'select', [100])
    tl.enqueue(tl.supervisor, 'merge', [0, 100], 1000)
    tl.process()

    assert tl.last_state() == ([1000], 1001, [101], 102)

    tl.enqueue(tl.supervisor, 'undo')
    tl.process()
    assert tl.last_state() == ([0], 1, [100], 101)

    tl.enqueue(tl.supervisor, 'redo')
    tl.process()
    assert tl.last_state() == ([1000], 1001, [101], 102)


def test_task_split(tl):
    tl.enqueue(tl.cluster_view, 'select', [0])
    tl.enqueue(tl.similarity_view, 'select', [100])
    tl.enqueue(tl.supervisor, 'split', [0, 100], [1000, 1001])
    tl.process()

    assert tl.last_state() == ([1000, 1001], 1002, None, None)


def test_task_move_1(tl):
    tl.enqueue(tl.cluster_view, 'select', [0])
    tl.enqueue(tl.supervisor, 'move', [0], 'good')
    tl.process()

    assert tl.last_state() == ([1], 2, None, None)


def test_task_move_best(tl):
    tl.enqueue(tl.cluster_view, 'select', [0])
    tl.enqueue(tl.similarity_view, 'select', [100])
    tl.enqueue(tl.supervisor, 'move', 'best', 'good')
    tl.process()

    assert tl.last_state() == ([1], 2, None, None)


def test_task_move_similar(tl):
    tl.enqueue(tl.cluster_view, 'select', [0])
    tl.enqueue(tl.similarity_view, 'select', [100])
    tl.enqueue(tl.supervisor, 'move', 'similar', 'good')
    tl.process()

    assert tl.last_state() == ([0], 1, [101], 102)


def test_task_move_all(tl):
    tl.enqueue(tl.cluster_view, 'select', [0])
    tl.enqueue(tl.similarity_view, 'select', [100])
    tl.enqueue(tl.supervisor, 'move', 'all', 'good')
    tl.process()

    assert tl.last_state() == ([1], 2, [101], 102)


#------------------------------------------------------------------------------
# Test cluster and similarity views
#------------------------------------------------------------------------------

@fixture
def data():
    _data = [{"id": i,
              "n_spikes": 100 - 10 * i,
              "quality": 100 - 10 * i,
              "group": {2: 'noise', 3: 'noise', 5: 'mua', 8: 'good'}.get(i, None),
              "is_masked": i in (2, 3, 5),
              } for i in range(10)]
    return _data


def test_cluster_view_1(qtbot, gui, data):
    cv = ClusterView(gui, data=data)
    _wait_until_table_ready(qtbot, cv)

    cv.sort_by('n_spikes', 'asc')
    _assert(cv.get_state, {'current_sort': ('n_spikes', 'asc')})

    cv.set_state({'current_sort': ('id', 'desc')})
    _assert(cv.get_state, {'current_sort': ('id', 'desc')})


def test_similarity_view_1(qtbot, gui, data):
    sv = SimilarityView(gui, data=data)
    _wait_until_table_ready(qtbot, sv)

    @connect(sender=sv)
    def on_request_similar_clusters(sender, cluster_id):
        return [{'id': id} for id in (100 + cluster_id, 110 + cluster_id, 102 + cluster_id)]

    sv.reset([5])
    _assert(sv.get_ids, [105, 115, 107])


def test_cluster_view_extra_columns(qtbot, gui, data):

    for cl in data:
        cl['my_metrics'] = cl['id'] * 1000

    cv = ClusterView(gui, data=data, columns=['id', 'n_spikes', 'quality', 'my_metrics'])
    _wait_until_table_ready(qtbot, cv)


#------------------------------------------------------------------------------
# Test ActionCreator
#------------------------------------------------------------------------------

def test_action_creator_1(qtbot, gui):
    ac = ActionCreator()
    ac.attach(gui)
    gui.show()


#------------------------------------------------------------------------------
# Test GUI component
#------------------------------------------------------------------------------

def _select(supervisor, cluster_ids, similar=None):
    supervisor.task_logger.enqueue(supervisor.cluster_view, 'select', cluster_ids)
    if similar is not None:
        supervisor.task_logger.enqueue(supervisor.similarity_view, 'select', similar)
    supervisor.task_logger.process()
    supervisor.block()
    supervisor.task_logger.show_history()

    assert supervisor.task_logger.last_state()[0] == cluster_ids
    assert supervisor.task_logger.last_state()[2] == similar


def _assert_selected(supervisor, sel):
    assert supervisor.selected == sel


def test_select(qtbot, supervisor):
    _select(supervisor, [30], [20])
    _assert_selected(supervisor, [30, 20])


def test_supervisor_cluster_metrics(
        qtbot, gui, cluster_ids, cluster_groups, quality, similarity, tempdir):
    spike_clusters = np.repeat(cluster_ids, 2)

    def my_metrics(cluster_id):
        return cluster_id ** 2

    cluster_metrics = {'my_metrics': my_metrics}

    mc = Supervisor(spike_clusters,
                    cluster_groups=cluster_groups,
                    cluster_metrics=cluster_metrics,
                    quality=quality,
                    similarity=similarity,
                    context=Context(tempdir),
                    )
    mc.attach(gui)
    b = Barrier()
    connect(b('cluster_view'), event='ready', sender=mc.cluster_view)
    connect(b('similarity_view'), event='ready', sender=mc.similarity_view)
    b.wait()

    assert 'my_metrics' in mc.columns


def test_supervisor_select_1(qtbot, supervisor):
    # WARNING: always use actions in tests, because this doesn't call
    # the supervisor method directly, but raises an event, enqueue the task,
    # and call TaskLogger.process() which handles the cascade of callbacks.
    supervisor.actions.select([0])
    supervisor.block()
    _assert_selected(supervisor, [0])
    supervisor.task_logger.show_history()


def test_supervisor_select_2(qtbot, supervisor):
    supervisor.actions.next_best()
    supervisor.block()
    _assert_selected(supervisor, [30])


def test_supervisor_select_order(qtbot, supervisor):
    _select(supervisor, [1, 0])
    _assert_selected(supervisor, [1, 0])
    _select(supervisor, [0, 1])
    _assert_selected(supervisor, [0, 1])


def test_supervisor_edge_cases(supervisor):

    # Empty selection at first.
    ae(supervisor.clustering.cluster_ids, [0, 1, 2, 10, 11, 20, 30])

    _select(supervisor, [0])

    supervisor.undo()
    supervisor.block()

    supervisor.redo()
    supervisor.block()

    # Merge.
    supervisor.merge()
    supervisor.block()
    _assert_selected(supervisor, [0])

    supervisor.merge([])
    supervisor.block()
    _assert_selected(supervisor, [0])

    supervisor.merge([10])
    supervisor.block()
    _assert_selected(supervisor, [0])

    # Split.
    supervisor.split([])
    supervisor.block()
    _assert_selected(supervisor, [0])

    # Move.
    supervisor.move('ignored', [])
    supervisor.block()

    supervisor.save()


def test_supervisor_skip(qtbot, gui, supervisor):

    # yield [0, 1, 2, 10, 11, 20, 30]
    # #      i, g, N,  i,  g,  N, N
    expected = [30, 20, 11, 2, 1]

    for clu in expected:
        supervisor.actions.next_best()
        supervisor.block()
        _assert_selected(supervisor, [clu])


def test_supervisor_merge_1(qtbot, supervisor):

    _select(supervisor, [30], [20])
    _assert_selected(supervisor, [30, 20])

    supervisor.actions.merge()
    supervisor.block()

    _assert_selected(supervisor, [31, 11])

    supervisor.actions.undo()
    supervisor.block()
    _assert_selected(supervisor, [30, 20])

    supervisor.actions.redo()
    supervisor.block()
    supervisor.task_logger.show_history()
    _assert_selected(supervisor, [31, 11])


def test_supervisor_merge_move(qtbot, supervisor):
    """Check that merge then move selects the next cluster in the original
    cluster view, not the updated cluster view."""

    _select(supervisor, [20, 11], [])
    _assert_selected(supervisor, [20, 11])

    supervisor.actions.merge()
    supervisor.block()
    _assert_selected(supervisor, [31])

    supervisor.actions.move('good', 'all')
    supervisor.block()
    _assert_selected(supervisor, [30])

    supervisor.actions.move('good', 'all')
    supervisor.block()
    _assert_selected(supervisor, [2])


def test_supervisor_split_0(supervisor):

    _select(supervisor, [1, 2])
    _assert_selected(supervisor, [1, 2])

    supervisor.actions.split([1, 2])
    supervisor.block()

    _assert_selected(supervisor, [31, 32, 33])

    supervisor.actions.undo()
    supervisor.block()
    _assert_selected(supervisor, [1, 2])

    supervisor.actions.redo()
    supervisor.block()
    _assert_selected(supervisor, [31, 32, 33])


def test_supervisor_split_1(supervisor):

    supervisor.actions.select([1, 2])
    supervisor.block()

    @connect(sender=supervisor)
    def on_request_split(sender):
        return [1, 2]

    supervisor.actions.split()
    supervisor.block()
    _assert_selected(supervisor, [31, 32, 33])


def test_supervisor_split_2(gui, quality, similarity):
    spike_clusters = np.array([0, 0, 1])

    supervisor = Supervisor(spike_clusters,
                            similarity=similarity,
                            )
    supervisor.attach(gui)

    b = Barrier()
    connect(b('cluster_view'), event='ready', sender=supervisor.cluster_view)
    connect(b('similarity_view'), event='ready', sender=supervisor.similarity_view)
    b.wait()

    supervisor.actions.split([0])
    supervisor.block()
    _assert_selected(supervisor, [2, 3])


def test_supervisor_state(tempdir, qtbot, gui, supervisor):

    cv = supervisor.cluster_view
    assert supervisor.state.cluster_view.current_sort == ('quality', 'desc')

    cv.sort_by('id')
    assert supervisor.state.cluster_view.current_sort == ('id', 'asc')

    cv.set_state({'current_sort': ('n_spikes', 'desc')})
    assert supervisor.state.cluster_view.current_sort == ('n_spikes', 'desc')


def test_supervisor_label(supervisor):

    _select(supervisor, [20])
    supervisor.label("my_field", 3.14)
    supervisor.block()

    supervisor.label("my_field", 1.23, cluster_ids=30)
    supervisor.block()

    supervisor.save()

    assert 'my_field' in supervisor.fields
    assert supervisor.get_labels('my_field')[20] == 3.14
    assert supervisor.get_labels('my_field')[30] == 1.23


def test_supervisor_move_1(supervisor):

    _select(supervisor, [20])
    _assert_selected(supervisor, [20])

    assert not supervisor.move('', '')

    supervisor.actions.move('noise', 'all')
    supervisor.block()
    _assert_selected(supervisor, [11])

    supervisor.actions.undo()
    supervisor.block()
    _assert_selected(supervisor, [20])

    supervisor.actions.redo()
    supervisor.block()
    _assert_selected(supervisor, [11])


def test_supervisor_move_2(supervisor):

    _select(supervisor, [20], [10])
    _assert_selected(supervisor, [20, 10])

    supervisor.actions.move('noise', 10)
    supervisor.block()
    _assert_selected(supervisor, [20, 2])

    supervisor.actions.undo()
    supervisor.block()
    _assert_selected(supervisor, [20, 10])

    supervisor.actions.redo()
    supervisor.block()
    _assert_selected(supervisor, [20, 2])


def test_supervisor_move_3(qtbot, supervisor):

    supervisor.actions.next()
    supervisor.block()
    _assert_selected(supervisor, [30])

    supervisor.actions.move_best_to_noise()
    supervisor.block()
    _assert_selected(supervisor, [20])

    supervisor.actions.move_best_to_mua()
    supervisor.block()
    _assert_selected(supervisor, [11])

    supervisor.actions.move_best_to_good()
    supervisor.block()
    _assert_selected(supervisor, [2])

    supervisor.cluster_meta.get('group', 30) == 'noise'
    supervisor.cluster_meta.get('group', 20) == 'mua'
    supervisor.cluster_meta.get('group', 11) == 'good'


def test_supervisor_move_4(supervisor):

    _select(supervisor, [30], [20])
    _assert_selected(supervisor, [30, 20])

    supervisor.actions.move_similar_to_noise()
    supervisor.block()
    _assert_selected(supervisor, [30, 11])

    supervisor.actions.move_similar_to_mua()
    supervisor.block()
    _assert_selected(supervisor, [30, 2])

    supervisor.actions.move_similar_to_good()
    supervisor.block()
    _assert_selected(supervisor, [30, 1])

    supervisor.cluster_meta.get('group', 20) == 'noise'
    supervisor.cluster_meta.get('group', 11) == 'mua'
    supervisor.cluster_meta.get('group', 2) == 'good'


def test_supervisor_move_5(supervisor):
    _select(supervisor, [30], [20])
    _assert_selected(supervisor, [30, 20])

    supervisor.actions.move_all_to_noise()
    supervisor.block()
    _assert_selected(supervisor, [11, 2])

    supervisor.actions.next()
    supervisor.block()
    _assert_selected(supervisor, [11, 1])

    supervisor.actions.move_all_to_mua()
    supervisor.block()
    _assert_selected(supervisor, [2])

    supervisor.actions.move_all_to_good()
    supervisor.block()
    _assert_selected(supervisor, [])

    supervisor.cluster_meta.get('group', 30) == 'noise'
    supervisor.cluster_meta.get('group', 20) == 'noise'

    supervisor.cluster_meta.get('group', 11) == 'mua'
    supervisor.cluster_meta.get('group', 10) == 'mua'

    supervisor.cluster_meta.get('group', 2) == 'good'
    supervisor.cluster_meta.get('group', 1) == 'good'


def test_supervisor_reset(qtbot, supervisor):

    supervisor.actions.select([10, 11])

    supervisor.actions.reset()
    supervisor.block()
    _assert_selected(supervisor, [30])

    supervisor.actions.next()
    supervisor.block()
    _assert_selected(supervisor, [30, 20])

    supervisor.actions.next()
    supervisor.block()
    _assert_selected(supervisor, [30, 11])

    supervisor.actions.previous()
    supervisor.block()
    _assert_selected(supervisor, [30, 20])


def test_supervisor_nav(qtbot, supervisor):

    supervisor.actions.reset()
    supervisor.block()
    _assert_selected(supervisor, [30])

    supervisor.actions.next_best()
    supervisor.block()
    _assert_selected(supervisor, [20])

    supervisor.actions.previous_best()
    supervisor.block()
    _assert_selected(supervisor, [30])
