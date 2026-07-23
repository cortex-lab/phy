"""Test GUI component."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# from contextlib import contextmanager

import sys

import numpy as np
from numpy.testing import assert_array_equal as ae
from phylib.utils import Bunch, connect, emit, unconnect
from pytest import fixture, raises

from phy.gui import GUI
from phy.gui.actions import _get_shortcut_string
from phy.gui.qt import QHeaderView, Qt, qInstallMessageHandler
from phy.gui.tests.test_widgets import _assert, _wait_until_table_ready
from phy.gui.widgets import Barrier
from phy.utils.context import Context

from .. import supervisor as _supervisor
from ..supervisor import (
    ActionCreator,
    ClusterView,
    SimilarityView,
    Supervisor,
    TaskLogger,
)


def handler(msg_type, msg_log_context, msg_string):
    pass


qInstallMessageHandler(handler)


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


@fixture
def gui(tempdir, qtbot):
    # NOTE: mock patch show box exec_
    _supervisor._show_box = lambda _: _

    gui = GUI(position=(200, 100), size=(500, 500), config_dir=tempdir)
    gui.set_default_actions()
    with qtbot.waitExposed(gui):
        gui.show()
    yield gui
    qtbot.wait(5)
    gui.close()
    del gui
    qtbot.wait(5)


@fixture
def supervisor(qtbot, gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir):
    spike_clusters = np.repeat(cluster_ids, 2)

    s = Supervisor(
        spike_clusters,
        cluster_groups=cluster_groups,
        cluster_labels=cluster_labels,
        similarity=similarity,
        context=Context(tempdir),
        sort=('id', 'desc'),
    )
    s.attach(gui)
    b = Barrier()
    connect(b('cluster_view'), event='ready', sender=s.cluster_view)
    connect(b('similarity_view'), event='ready', sender=s.similarity_view)
    b.wait()
    return s


# ------------------------------------------------------------------------------
# Test tasks
# ------------------------------------------------------------------------------


@fixture
def tl():
    class MockClusterView:
        _selected = [0]

        def select(self, cl, callback=None, **kwargs):
            self._selected = cl
            callback({'selected': cl, 'next': cl[-1] + 1})

        def next(self, callback=None):
            callback({'selected': [self._selected[-1] + 1], 'next': self._selected[-1] + 2})

        def previous(self, callback=None):  # pragma: no cover
            callback({'selected': [self._selected[-1] - 1], 'next': self._selected[-1]})

    class MockSimilarityView(MockClusterView):
        pass

    class MockSupervisor:
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

    assert tl.last_state() == ([1000], 1001, None, None)

    tl.enqueue(tl.supervisor, 'undo')
    tl.process()
    assert tl.last_state() == ([0], 1, [100], 101)

    tl.enqueue(tl.supervisor, 'redo')
    tl.process()
    assert tl.last_state() == ([1000], 1001, None, None)


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


# ------------------------------------------------------------------------------
# Test cluster and similarity views
# ------------------------------------------------------------------------------


@fixture
def data():
    _data = [
        {
            'id': i,
            'n_spikes': 100 - 10 * i,
            'group': {2: 'noise', 3: 'noise', 5: 'mua', 8: 'good'}.get(i),
            'is_masked': i in (2, 3, 5),
        }
        for i in range(10)
    ]
    return _data


def test_cluster_view_1(qtbot, gui, data):
    cv = ClusterView(gui, data=data)
    _wait_until_table_ready(qtbot, cv)
    assert cv.debouncer.delay == 50

    cv.sort_by('n_spikes', 'asc')
    cv.select([1])
    qtbot.wait(10)
    assert cv.state == {'current_sort': ('n_spikes', 'asc'), 'selected': [1]}

    cv.set_state({'current_sort': ('id', 'desc'), 'selected': [2]})
    assert cv.state == {'current_sort': ('id', 'desc'), 'selected': [2]}


def test_cluster_view_formats_spike_counts(qtbot, gui):
    cv = ClusterView(gui, data=[{'id': 1, 'n_spikes': 1234567}])
    _wait_until_table_ready(qtbot, cv)

    index = cv._proxy.index(0, cv.columns.index('n_spikes'))
    assert index.data(Qt.DisplayRole) == '1,234,567'
    assert index.data(Qt.EditRole) == 1234567


def test_similarity_view_1(qtbot, gui):
    sv = SimilarityView(gui)
    _wait_until_table_ready(qtbot, sv)
    assert sv.debouncer.delay == 50

    @connect(sender=sv)
    def on_request_similar_clusters(sender, cluster_id):
        if cluster_id == 5:
            return [
                {'id': 105, 'n_spikes': int('9' * 100), 'similarity': 0.9},
                {'id': 115, 'n_spikes': int('8' * 90), 'similarity': 0.8},
                {'id': 107, 'n_spikes': int('7' * 80), 'similarity': 0.7},
            ]
        return [
            {'id': id, 'n_spikes': n_spikes, 'similarity': similarity}
            for id, n_spikes, similarity in (
                (106, 3, 0.3),
                (116, 2, 0.2),
                (108, 1, 0.1),
            )
        ]

    header = sv.table_view.horizontalHeader()
    vertical_header = sv.table_view.verticalHeader()
    header_only_widths = [header.sectionSize(i) for i in range(header.count())]

    sv.reset([5])
    qtbot.wait(1)
    _assert(sv.get_ids, [105, 115, 107])
    fitted_widths = [header.sectionSize(i) for i in range(header.count())]
    resize_modes = [header.sectionResizeMode(i) for i in range(header.count())]
    row_heights = [vertical_header.sectionSize(i) for i in range(3)]
    assert any(after > before for before, after in zip(header_only_widths, fitted_widths))
    assert resize_modes == [QHeaderView.Interactive] * header.count()

    sv.reset([6])
    qtbot.wait(1)
    _assert(sv.get_ids, [106, 116, 108])
    assert [header.sectionSize(i) for i in range(header.count())] == fitted_widths
    assert [header.sectionResizeMode(i) for i in range(header.count())] == resize_modes
    assert [vertical_header.sectionSize(i) for i in range(3)] == row_heights

    unconnect(on_request_similar_clusters)
    sv.reset([7])
    qtbot.wait(1)
    _assert(sv.get_ids, [])
    assert [header.sectionSize(i) for i in range(header.count())] == fitted_widths
    assert [header.sectionResizeMode(i) for i in range(header.count())] == resize_modes


def test_cluster_view_extra_columns(qtbot, gui, data):
    for cl in data:
        cl['my_metrics'] = cl['id'] * 1000

    cv = ClusterView(gui, data=data, columns=['id', 'n_spikes', 'my_metrics'])
    _wait_until_table_ready(qtbot, cv)


# ------------------------------------------------------------------------------
# Test ActionCreator
# ------------------------------------------------------------------------------


def test_action_creator_1(qtbot, gui):
    ac = ActionCreator()
    ac.attach(gui)
    gui.show()


# ------------------------------------------------------------------------------
# Test GUI component
# ------------------------------------------------------------------------------


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


def test_block_flushes_pending_selections(qtbot, supervisor):
    supervisor.cluster_view.debouncer.delay = 60_000
    supervisor.similarity_view.debouncer.delay = 60_000

    supervisor.select([30])
    supervisor.block()
    assert supervisor.selected_clusters == [30]

    # This selection falls inside the debounce interval and remains pending until block().
    supervisor.select([20])
    assert supervisor.selected_clusters == [30]
    supervisor.block()
    assert supervisor.selected_clusters == [20]

    similar_cluster_id = supervisor.similarity_view.get_ids()[0]
    supervisor.similarity_view.select([similar_cluster_id])
    supervisor.block()
    assert supervisor.selected_similar == [similar_cluster_id]

    next_similar_cluster_id = supervisor.similarity_view.get_ids()[1]
    supervisor.similarity_view.select([next_similar_cluster_id])
    assert supervisor.selected_similar == [similar_cluster_id]
    supervisor.block()
    assert supervisor.selected_similar == [next_similar_cluster_id]


def test_supervisor_busy(qtbot, supervisor):
    _select(supervisor, [30], [20])

    o = object()

    emit('is_busy', o, True)
    assert supervisor._is_busy

    # The action fails while the supervisor is busy.
    emit('action', supervisor.action_creator, 'merge')

    emit('is_busy', o, False)
    assert not supervisor._is_busy

    # The action succeeds because the supervisor is no longer busy.
    emit('action', supervisor.action_creator, 'merge')
    supervisor.block()
    assert not supervisor._is_busy


def test_supervisor_cluster_metrics(qtbot, gui, cluster_ids, cluster_groups, similarity, tempdir):
    spike_clusters = np.repeat(cluster_ids, 2)

    def my_metrics(cluster_id):
        return cluster_id**2

    cluster_metrics = {'my_metrics': my_metrics}

    mc = Supervisor(
        spike_clusters,
        cluster_groups=cluster_groups,
        cluster_metrics=cluster_metrics,
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
    supervisor.select_actions.select([0])
    supervisor.block()
    _assert_selected(supervisor, [0])
    supervisor.task_logger.show_history()


def test_supervisor_select_2(qtbot, supervisor):
    supervisor.select_actions.next_best()
    supervisor.block()
    _assert_selected(supervisor, [30])


def test_supervisor_select_order(qtbot, supervisor):
    _select(supervisor, [1, 0])
    _assert_selected(supervisor, [1, 0])
    _select(supervisor, [0, 1])
    _assert_selected(supervisor, [0, 1])


def test_supervisor_select_first_similar(qtbot, supervisor, gui):
    _select(supervisor, [30])
    similarity_view = supervisor.similarity_view

    similarity_view.sort_by('id', 'asc')
    similarity_view.filter('id >= 10')
    navigable_ids = similarity_view.get_navigable_ids()

    # The prompted variant updates the preference and selects eligible rows in visible order.
    supervisor.select_actions.select_n_similar(2)
    supervisor.block()
    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == navigable_ids[:2]
    assert supervisor.n_similar_clusters_to_select == 2

    # The shortcut variant uses the saved preference and replaces the similar selection.
    similarity_view.sort_by('id', 'desc')
    navigable_ids = similarity_view.get_navigable_ids()
    control_modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier
    qtbot.keyClick(gui, Qt.Key_Space, control_modifier)
    supervisor.block()
    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == navigable_ids[:2]

    # Selecting more rows than are available is safe.
    supervisor.select_actions.select_n_similar(100)
    supervisor.block()
    assert supervisor.selected_similar == navigable_ids

    # The preference is stored in global GUI state.
    supervisor._save_gui_state(gui)
    assert gui.state['n_similar_clusters_to_select'] == 100


def test_filter_release_restores_space_shortcut(qtbot, supervisor, gui):
    _select(supervisor, [30], [2])
    similarity_view = supervisor.similarity_view
    similarity_view.sort_by('id', 'asc')

    qtbot.mouseClick(similarity_view.filter_edit, Qt.LeftButton)
    assert similarity_view.filter_edit.hasFocus()
    qtbot.keyClick(similarity_view.filter_edit, Qt.Key_Return)
    assert not similarity_view.filter_edit.hasFocus()

    qtbot.keyClick(gui, Qt.Key_Space)
    supervisor.block()
    assert supervisor.selected_similar == [11]


def test_supervisor_skip_masked_navigation_and_selection(supervisor):
    assert supervisor.skip_masked_clusters is True
    assert supervisor.cluster_view.skip_masked is True
    assert supervisor.similarity_view.skip_masked is True

    # Cluster-view traversal skips MUA cluster 10 by default.
    _select(supervisor, [11])
    supervisor.select_actions.next_best()
    supervisor.block()
    assert supervisor.selected_clusters == [2]

    # Similarity-view traversal observes the same policy.
    _select(supervisor, [30], [2])
    supervisor.similarity_view.sort_by('id', 'asc')
    supervisor.select_actions.next()
    supervisor.block()
    assert supervisor.selected_similar == [11]

    # Direct selection remains unrestricted.
    supervisor.select_actions.select([10])
    supervisor.block()
    assert supervisor.selected_clusters == [10]

    supervisor.set_skip_masked_clusters(False)
    assert supervisor.cluster_view.skip_masked is False
    assert supervisor.similarity_view.skip_masked is False

    _select(supervisor, [11])
    supervisor.select_actions.next_best()
    supervisor.block()
    assert supervisor.selected_clusters == [10]

    _select(supervisor, [30], [2])
    supervisor.similarity_view.sort_by('id', 'asc')
    supervisor.select_actions.next()
    supervisor.block()
    assert supervisor.selected_similar == [10]


def test_supervisor_select_first_similar_obeys_skip_masked_policy(supervisor):
    _select(supervisor, [30])
    supervisor.similarity_view.sort_by('id', 'asc')
    visible_ids = supervisor.similarity_view.get_ids()
    assert visible_ids[:2] == [0, 1]

    supervisor.select_actions.select_n_similar(2)
    supervisor.block()
    assert supervisor.selected_similar == [1, 2]

    supervisor.set_skip_masked_clusters(False)
    supervisor.select_actions.select_n_similar(2)
    supervisor.block()
    assert supervisor.selected_similar == [0, 1]


def test_supervisor_select_first_similar_empty(supervisor):
    _select(supervisor, [30])
    supervisor.similarity_view.filter('id > 1000')
    supervisor.select_actions.select_n_similar(3)
    supervisor.block()
    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == []


def test_supervisor_select_first_similar_config(gui, cluster_ids, similarity):
    gui.state['n_similar_clusters_to_select'] = 4
    supervisor = Supervisor(
        np.repeat(cluster_ids, 2),
        similarity=similarity,
        n_similar_clusters_to_select=2,
    )
    supervisor.attach(gui)
    assert supervisor.n_similar_clusters_to_select == 4

    shortcut = supervisor.select_actions.get('select_first_similar').shortcut()
    expected_shortcut = 'meta+space' if sys.platform == 'darwin' else 'ctrl+space'
    assert _get_shortcut_string(shortcut) == expected_shortcut

    with raises(ValueError, match='positive integer'):
        supervisor.select_first_similar(0)
    with raises(ValueError, match='positive integer'):
        supervisor.select_first_similar(1.5)


def test_supervisor_skip_masked_config_menu_and_state(gui, cluster_ids, similarity):
    gui.state['skip_masked_clusters'] = False
    supervisor = Supervisor(
        np.repeat(cluster_ids, 2),
        similarity=similarity,
        skip_masked_clusters=True,
    )
    supervisor.attach(gui)

    # Saved GUI state overrides the constructor default and initializes both views and action.
    assert supervisor.skip_masked_clusters is False
    assert supervisor.cluster_view.skip_masked is False
    assert supervisor.similarity_view.skip_masked is False
    action = supervisor.select_actions.get('skip_noise_and_mua')
    assert not action.isChecked()

    action.trigger()
    supervisor.block()
    assert supervisor.skip_masked_clusters is True
    assert supervisor.cluster_view.skip_masked is True
    assert supervisor.similarity_view.skip_masked is True
    assert action.isChecked()

    supervisor._save_gui_state(gui)
    assert gui.state['skip_masked_clusters'] is True


def test_supervisor_skip_masked_constructor_and_invalid_state(gui, cluster_ids, similarity):
    gui.state['skip_masked_clusters'] = 'invalid'
    supervisor = Supervisor(
        np.repeat(cluster_ids, 2),
        similarity=similarity,
        skip_masked_clusters=False,
    )
    supervisor.attach(gui)

    # Invalid saved state is ignored, leaving the constructor preference in force.
    assert supervisor.skip_masked_clusters is False
    assert supervisor.cluster_view.skip_masked is False
    assert supervisor.similarity_view.skip_masked is False
    assert not supervisor.select_actions.get('skip_noise_and_mua').isChecked()


def test_supervisor_promote_similar_with_control_right_click(qtbot, supervisor):
    _select(supervisor, [10, 30], [20, 11, 1])
    similarity_view = supervisor.similarity_view
    similarity_view.sort_by('id', 'asc')
    similarity_view.filter('id >= 1')

    index = similarity_view._proxy_index_for_id(11)
    pos = similarity_view.table_view.visualRect(index).center()
    qtbot.mouseClick(similarity_view.table_view.viewport(), Qt.RightButton, pos=pos)
    supervisor.block()
    assert supervisor.selected_clusters == [10, 30]
    assert supervisor.selected_similar == [20, 11, 1]

    control_modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier
    qtbot.mouseClick(
        similarity_view.table_view.viewport(), Qt.RightButton, control_modifier, pos=pos
    )
    supervisor.block()

    assert supervisor.selected_clusters == [10, 11, 30]
    assert supervisor.selected_similar == [20, 1]
    assert supervisor.selected == [10, 11, 30, 20, 1]
    assert 11 not in similarity_view.get_ids()


def test_supervisor_promote_unselected_similar_with_control_right_click(qtbot, supervisor):
    _select(supervisor, [30], [20, 11])
    similarity_view = supervisor.similarity_view

    index = similarity_view._proxy_index_for_id(1)
    pos = similarity_view.table_view.visualRect(index).center()
    control_modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier
    qtbot.mouseClick(
        similarity_view.table_view.viewport(), Qt.RightButton, control_modifier, pos=pos
    )
    supervisor.block()

    assert supervisor.selected_clusters == [1, 30]
    assert supervisor.selected_similar == [20, 11]


def test_supervisor_demote_cluster_with_control_right_click(qtbot, supervisor):
    _select(supervisor, [10, 30], [20, 11])
    cluster_view = supervisor.cluster_view
    control_modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier

    index = cluster_view._proxy_index_for_id(10)
    pos = cluster_view.table_view.visualRect(index).center()
    qtbot.mouseClick(cluster_view.table_view.viewport(), Qt.RightButton, control_modifier, pos=pos)
    supervisor.block()

    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == [20, 11, 10]
    assert supervisor.selected == [30, 20, 11, 10]

    index = cluster_view._proxy_index_for_id(30)
    pos = cluster_view.table_view.visualRect(index).center()
    qtbot.mouseClick(cluster_view.table_view.viewport(), Qt.RightButton, control_modifier, pos=pos)
    supervisor.block()

    # Keep one cluster as the similarity reference.
    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == [20, 11, 10]

    index = cluster_view._proxy_index_for_id(1)
    pos = cluster_view.table_view.visualRect(index).center()
    qtbot.mouseClick(cluster_view.table_view.viewport(), Qt.RightButton, control_modifier, pos=pos)
    supervisor.block()

    # Rows outside the Cluster View selection cannot be transferred.
    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == [20, 11, 10]


def test_supervisor_control_left_click_toggles_selection_in_each_view(qtbot, supervisor):
    _select(supervisor, [10, 30], [20])
    control_modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier

    cluster_view = supervisor.cluster_view
    index = cluster_view._proxy_index_for_id(10)
    pos = cluster_view.table_view.visualRect(index).center()
    qtbot.mouseClick(cluster_view.table_view.viewport(), Qt.LeftButton, control_modifier, pos=pos)
    supervisor.block()

    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == []

    similarity_view = supervisor.similarity_view
    index = similarity_view._proxy_index_for_id(20)
    pos = similarity_view.table_view.visualRect(index).center()
    qtbot.mouseClick(
        similarity_view.table_view.viewport(), Qt.LeftButton, control_modifier, pos=pos
    )
    supervisor.block()

    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == [20]

    qtbot.mouseClick(
        similarity_view.table_view.viewport(), Qt.LeftButton, control_modifier, pos=pos
    )
    supervisor.block()

    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == []


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


def test_supervisor_save(qtbot, gui, supervisor):
    emit('request_save', gui)


def test_supervisor_skip(qtbot, gui, supervisor):
    # yield [0, 1, 2, 10, 11, 20, 30]
    # #      i, g, N,  i,  g,  N, N
    expected = [30, 20, 11, 2, 1]

    for clu in expected:
        supervisor.select_actions.next_best()
        supervisor.block()
        _assert_selected(supervisor, [clu])


def test_supervisor_sort(qtbot, supervisor):
    supervisor.sort('id', 'desc')
    qtbot.wait(50)
    assert supervisor.state.cluster_view.current_sort == ('id', 'desc')

    supervisor.select_actions.sort_by_n_spikes()
    qtbot.wait(50)
    assert supervisor.state.cluster_view.current_sort == ('n_spikes', 'desc')


def test_supervisor_filter(qtbot, supervisor):
    supervisor.filter('5 <= id && id <= 20')
    qtbot.wait(50)
    _cl = []
    supervisor.cluster_view.get_ids(lambda cluster_ids: _cl.extend(cluster_ids))
    qtbot.wait(50)
    assert _cl == [20, 11, 10]
    supervisor.clear_filter()
    qtbot.wait(50)


def test_supervisor_merge_1(qtbot, supervisor):
    _select(supervisor, [30], [20])
    _assert_selected(supervisor, [30, 20])

    supervisor.actions.merge()
    supervisor.block()

    _assert_selected(supervisor, [31])

    supervisor.actions.undo()
    supervisor.block()
    _assert_selected(supervisor, [30, 20])

    supervisor.actions.redo()
    supervisor.block()
    supervisor.task_logger.show_history()
    _assert_selected(supervisor, [31])

    assert supervisor.is_dirty()


def test_supervisor_merge_event(qtbot, supervisor):
    _select(supervisor, [30], [20])

    _l = []

    @connect(sender=supervisor)
    def on_select(sender, cluster_ids):
        _l.append(cluster_ids)

    supervisor.actions.merge()
    supervisor.block()

    # After a merge, there should be only one select event.
    assert len(_l) == 1


def test_supervisor_merge_batches_table_fitting(monkeypatch, supervisor):
    _select(supervisor, [30], [20])
    fit_calls = {'cluster': 0, 'similarity': 0}

    monkeypatch.setattr(
        supervisor.cluster_view,
        '_fit_columns',
        lambda: fit_calls.__setitem__('cluster', fit_calls['cluster'] + 1),
    )
    monkeypatch.setattr(
        supervisor.similarity_view,
        '_fit_columns',
        lambda: fit_calls.__setitem__('similarity', fit_calls['similarity'] + 1),
    )

    supervisor.actions.merge()
    supervisor.block()

    assert fit_calls == {'cluster': 1, 'similarity': 1}


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


def test_supervisor_split_0(qtbot, supervisor):
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
    supervisor.select_actions.select([1, 2])
    supervisor.block()

    @connect(sender=supervisor)
    def on_request_split(sender):
        return [1, 2]

    supervisor.actions.split()
    supervisor.block()
    _assert_selected(supervisor, [31, 32, 33])


def test_supervisor_split_2(gui, similarity):
    spike_clusters = np.array([0, 0, 1])

    supervisor = Supervisor(spike_clusters, similarity=similarity)
    supervisor.attach(gui)

    b = Barrier()
    connect(b('cluster_view'), event='ready', sender=supervisor.cluster_view)
    connect(b('similarity_view'), event='ready', sender=supervisor.similarity_view)
    b.wait()

    supervisor.actions.split([0])
    supervisor.block()
    _assert_selected(supervisor, [2, 3])


def test_supervisor_state(tempdir, qtbot, gui, supervisor):
    supervisor.select(1)

    cv = supervisor.cluster_view
    assert supervisor.state.cluster_view.current_sort == ('id', 'desc')
    assert supervisor.state.cluster_view.selected == [1]

    cv.sort_by('id')
    assert supervisor.state.cluster_view.current_sort == ('id', 'asc')

    cv.set_state({'current_sort': ('n_spikes', 'desc')})
    assert supervisor.state.cluster_view.current_sort == ('n_spikes', 'desc')

    cv.sort_by('id', 'desc')
    assert supervisor.shown_cluster_ids == [30, 20, 11, 10, 2, 1, 0]


def test_supervisor_label(supervisor):
    _select(supervisor, [20])
    supervisor.label('my_field', 3.14)
    supervisor.block()

    supervisor.label('my_field', 1.23, cluster_ids=30)
    supervisor.block()

    assert 'my_field' in supervisor.fields
    assert supervisor.get_labels('my_field')[20] == 3.14
    assert supervisor.get_labels('my_field')[30] == 1.23


def test_supervisor_label_cluster_1(supervisor):
    _select(supervisor, [20, 30])
    supervisor.label('my_field', 3.14)
    supervisor.block()

    # Same value for the old clusters.
    l = supervisor.get_labels('my_field')
    assert l[20] == l[30] == 3.14

    up = supervisor.merge()
    supervisor.block()

    assert supervisor.get_labels('my_field')[up.added[0]] == 3.14


def test_supervisor_label_cluster_2(supervisor):
    _select(supervisor, [20])

    supervisor.label('my_field', 3.14)
    supervisor.block()

    # One of the parents.
    l = supervisor.get_labels('my_field')
    assert l[20] == 3.14
    assert l[30] is None

    up = supervisor.merge([20, 30])
    supervisor.block()

    assert supervisor.get_labels('my_field')[up.added[0]] == 3.14


def test_supervisor_label_cluster_3(supervisor):
    # Conflict: largest cluster wins.
    _select(supervisor, [20, 30])
    supervisor.label('my_field', 3.14)
    supervisor.block()

    # Create merged cluster from 20 and 30.
    up = supervisor.merge()
    new = up.added[0]
    supervisor.block()

    # It got the label of its parents.
    assert supervisor.get_labels('my_field')[new] == 3.14

    # Now, we label a smaller cluster.
    supervisor.label('my_field', 2.718, cluster_ids=[10])

    # We merge the large and small cluster together.
    up = supervisor.merge(up.added + [10])
    supervisor.block()

    # The new cluster should have the value of the first, merged big cluster, i.e. 3.14.
    assert supervisor.get_labels('my_field')[up.added[0]] == 3.14


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


def test_supervisor_move_undo_restores_table_group(supervisor):
    cluster_id = 0
    original_group = supervisor.cluster_meta.get('group', cluster_id)
    _select(supervisor, [cluster_id])

    supervisor.actions.move('good', cluster_id)
    supervisor.block()
    assert supervisor.cluster_view._model.row_by_id(cluster_id)['group'] == 'good'

    supervisor.actions.undo()
    supervisor.block()
    assert supervisor.cluster_meta.get('group', cluster_id) == original_group
    assert supervisor.cluster_view._model.row_by_id(cluster_id)['group'] == original_group


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
    supervisor.select_actions.next()
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

    assert supervisor.cluster_meta.get('group', 30) == 'noise'
    assert supervisor.cluster_meta.get('group', 20) == 'mua'
    assert supervisor.cluster_meta.get('group', 11) == 'good'


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

    assert supervisor.cluster_meta.get('group', 20) == 'noise'
    assert supervisor.cluster_meta.get('group', 11) == 'mua'
    assert supervisor.cluster_meta.get('group', 2) == 'good'


def test_supervisor_move_5(supervisor):
    _select(supervisor, [30], [20])
    _assert_selected(supervisor, [30, 20])

    supervisor.actions.move_all_to_noise()
    supervisor.block()
    _assert_selected(supervisor, [11, 2])

    supervisor.select_actions.next()
    supervisor.block()
    _assert_selected(supervisor, [11, 1])

    supervisor.actions.move_all_to_mua()
    supervisor.block()
    _assert_selected(supervisor, [2])

    supervisor.actions.move_all_to_good()
    supervisor.block()
    _assert_selected(supervisor, [])

    assert supervisor.cluster_meta.get('group', 30) == 'noise'
    assert supervisor.cluster_meta.get('group', 20) == 'noise'

    assert supervisor.cluster_meta.get('group', 11) == 'mua'
    assert supervisor.cluster_meta.get('group', 10) == 'mua'

    assert supervisor.cluster_meta.get('group', 2) == 'good'
    assert supervisor.cluster_meta.get('group', 1) == 'mua'


def test_supervisor_reset(qtbot, supervisor):
    supervisor.select_actions.select([10, 11])
    supervisor.block()
    _assert_selected(supervisor, [10, 11])

    supervisor.select_actions.reset_wizard()
    supervisor.block()
    _assert_selected(supervisor, [30])

    supervisor.select_actions.next()
    supervisor.block()
    _assert_selected(supervisor, [30, 20])

    supervisor.select_actions.next()
    supervisor.block()
    _assert_selected(supervisor, [30, 11])

    supervisor.select_actions.previous()
    supervisor.block()
    _assert_selected(supervisor, [30, 20])

    supervisor.select_actions.unselect_similar()
    supervisor.block()
    _assert_selected(supervisor, [30])


def test_supervisor_nav(qtbot, supervisor):
    supervisor.select_actions.reset_wizard()
    supervisor.block()
    _assert_selected(supervisor, [30])

    supervisor.select_actions.next_best()
    supervisor.block()
    _assert_selected(supervisor, [20])

    supervisor.select_actions.previous_best()
    supervisor.block()
    _assert_selected(supervisor, [30])

    supervisor.select_actions.first()
    qtbot.wait(100)
    _assert_selected(supervisor, [30])

    supervisor.select_actions.last()
    qtbot.wait(100)
    _assert_selected(supervisor, [1])


def test_supervisor_wizard_primary_navigation_clears_similar(supervisor):
    supervisor.cluster_view.debouncer.delay = 60_000
    supervisor.similarity_view.debouncer.delay = 60_000

    supervisor.select_actions.reset_wizard()
    supervisor.block()
    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == []

    supervisor.select_actions.next()
    supervisor.block()
    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == [20]

    supervisor.select_actions.next_best()
    supervisor.block()
    assert supervisor.selected_clusters == [20]
    assert supervisor.selected_similar == []
    assert supervisor.similarity_view.get_selected_ids() == []

    supervisor.select_actions.previous_best()
    supervisor.block()
    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == []
