"""Test GUI component."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# from contextlib import contextmanager

import sys
from timeit import default_timer

import numpy as np
from numpy.testing import assert_array_equal as ae
from phylib.utils import Bunch, connect, emit, unconnect
from phylib.utils.event import _EVENT
from pytest import fixture, mark, raises

from phy.gui import GUI
from phy.gui.actions import _get_shortcut_string
from phy.gui.qt import QAbstractItemView, QHeaderView, Qt, qInstallMessageHandler
from phy.gui.tests.test_widgets import _assert, _wait_until_table_ready
from phy.gui.widgets import Barrier
from phy.utils.color import selected_cluster_color
from phy.utils.context import Context

from .. import supervisor as _supervisor
from .._propositions import (
    MergePropositionController,
    PropositionStatus,
    decode_curation_mapping,
)
from ..supervisor import (
    ActionCreator,
    ClusterView,
    MergeView,
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
        post_actions = None

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

        def fail(self):
            raise RuntimeError('task failed')

        def _select_after_merge(self, output, selection_before, **kwargs):
            self.post_actions = ('merge', output, selection_before, kwargs)

        def _select_after_split(self, output):
            self.post_actions = ('split', output)

        def _select_after_move(self, selection_before, cluster_ids):
            self.post_actions = ('move', selection_before, cluster_ids)

    out = TaskLogger(MockClusterView(), MockSimilarityView(), MockSupervisor())

    return out


def test_task_logger_runs_callback_compatible_table_task(tl):
    tl.enqueue(tl.cluster_view, 'select', [0])
    tl.process()
    assert tl._history[-1][1] == 'select'
    assert tl._history[-1][-1] == {'selected': [0], 'next': 1}


def test_task_logger_delegates_merge_follow_up(tl):
    tl.enqueue(tl.supervisor, 'merge', [0, 100], 1000)
    tl.process()

    name, output, selection_before, kwargs = tl.supervisor.post_actions
    assert name == 'merge'
    assert output.added == [1000]
    assert selection_before is None
    assert kwargs['auto_select'] is False


def test_task_logger_delegates_split_follow_up(tl):
    tl.enqueue(tl.supervisor, 'split', [0, 100], [1000, 1001])
    tl.process()

    name, output = tl.supervisor.post_actions
    assert name == 'split'
    assert output.added == [1000, 1001]


def test_task_logger_delegates_move_follow_up(tl):
    tl.enqueue(tl.supervisor, 'move', [0], 'good')
    tl.process()

    assert tl.supervisor.post_actions == ('move', None, [0])


def test_task_logger_cleans_temporary_callback_and_queue_after_failure(tl):
    tl.enqueue(tl.supervisor, 'fail')
    with raises(RuntimeError, match='task failed'):
        tl.process()

    assert not tl._processing
    assert tl._queue == []
    history = list(tl._history)
    emit('cluster', tl.supervisor, Bunch())
    assert tl._history == history


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
    assert cv.state == {
        'current_sort': ('n_spikes', 'asc'),
        'selected': [1],
        'column_order': ['id', 'n_spikes'],
    }

    cv.set_state({'current_sort': ('id', 'desc'), 'selected': [2]})
    assert cv.state == {
        'current_sort': ('id', 'desc'),
        'selected': [2],
        'column_order': ['id', 'n_spikes'],
    }


def test_cluster_view_plain_right_click_reports_unselected_row_without_selecting_it(
    qtbot, gui, data
):
    cv = ClusterView(gui, data=data)
    _wait_until_table_ready(qtbot, cv)
    cv.select([1])
    qtbot.wait(10)

    clicked = []

    @connect(sender=cv)
    def on_row_right_click(sender, cluster_id):
        clicked.append(cluster_id)

    index = cv._proxy_index_for_id(2)
    pos = cv.table_view.visualRect(index).center()
    qtbot.mouseClick(cv.table_view.viewport(), Qt.RightButton, pos=pos)

    assert clicked == [2]
    assert cv.get_selected_ids() == [1]

    unconnect(on_row_right_click)


def test_cluster_view_formats_spike_counts(qtbot, gui):
    cv = ClusterView(gui, data=[{'id': 1, 'n_spikes': 1234567}])
    _wait_until_table_ready(qtbot, cv)

    index = cv._proxy.index(0, cv.columns.index('n_spikes'))
    assert index.data(Qt.DisplayRole) == '1,234,567'
    assert index.data(Qt.EditRole) == 1234567


def test_cluster_view_formats_multiple_values(qtbot, gui):
    cv = ClusterView(
        gui,
        data=[{'id': 1, 'n_spikes': 10, 'tags': ['tag_a', 'tag_b']}],
        columns=['tags'],
    )
    _wait_until_table_ready(qtbot, cv)

    index = cv._proxy.index(0, cv.columns.index('tags'))
    assert index.data(Qt.DisplayRole) == 'tag_a, tag_b'


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
    assert supervisor.selected_clusters == cluster_ids
    assert supervisor.selected_similar == (similar or [])


def _assert_selected(supervisor, sel):
    assert supervisor.selected == sel


def test_select(qtbot, supervisor):
    _select(supervisor, [30], [20])
    _assert_selected(supervisor, [30, 20])
    assert supervisor.selection.state.cluster_ids == (30,)
    assert supervisor.selection.state.similar_ids == (20,)
    assert supervisor.selection.state.reference_id == 30
    assert supervisor.selection.state.presentation_order == (30, 20)


def test_supervisor_selection_is_independent_from_task_log(supervisor):
    _select(supervisor, [30], [20])

    supervisor.task_logger._history.clear()

    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == [20]
    assert supervisor.selected == [30, 20]


def test_supervisor_merge_mode_lifecycle_restores_entry_state(supervisor):
    _select(supervisor, [10, 30], [20, 11])
    entry = supervisor.selection.snapshot()
    supervisor.cluster_view.filter('id >= 10')
    supervisor.similarity_view.filter('id >= 1')
    context = supervisor._workflow_context()
    events = []

    @connect(sender=supervisor)
    def on_select(sender, cluster_ids):
        events.append(cluster_ids)

    state = supervisor.toggle_merge_mode()

    assert state.is_merge_mode
    assert supervisor.selected_merge == [10, 30, 20, 11]
    assert supervisor.selected_clusters == []
    assert supervisor.selected_similar == []
    assert supervisor.selected == [10, 30, 20, 11]
    assert isinstance(supervisor.merge_view, MergeView)
    assert supervisor.merge_view.get_ids() == [10, 30, 20, 11]
    assert supervisor.merge_view.dock.get_widget('cancel_merge_mode') is not None
    assert supervisor.cluster_view.isEnabled()
    assert supervisor.cluster_view._interaction_blocked
    assert supervisor.cluster_view._interaction_overlay.isVisible()
    assert 'Press V' in supervisor.cluster_view._interaction_overlay.text()
    assert events == []

    supervisor.similarity_view.filter('id < 20')
    merge_view = supervisor.merge_view
    supervisor.toggle_merge_mode()

    assert supervisor.selection.state == entry
    assert supervisor.merge_view is merge_view
    assert merge_view in supervisor.gui.views
    assert merge_view.dock.isHidden()
    assert supervisor.cluster_view.isEnabled()
    assert not supervisor.cluster_view._interaction_blocked
    assert supervisor._workflow_context() == context
    assert events == []
    unconnect(on_select)


def test_closing_merge_view_restores_original_table_rows(qtbot, supervisor):
    _select(supervisor, [10, 30], [20, 11])
    cluster_rows = supervisor.cluster_view.get_ids()
    similarity_rows = supervisor.similarity_view.get_ids()

    supervisor.toggle_merge_mode()
    assert 20 not in supervisor.similarity_view.get_ids()
    assert 11 not in supervisor.similarity_view.get_ids()
    merge_view = supervisor.merge_view

    supervisor.merge_view.dock.close()
    qtbot.wait(10)

    assert not supervisor.selection.state.is_merge_mode
    assert supervisor.merge_view is merge_view
    assert merge_view in supervisor.gui.views
    assert merge_view.dock.isHidden()
    assert supervisor.cluster_view.get_ids() == cluster_rows
    assert supervisor.similarity_view.get_ids() == similarity_rows
    assert supervisor.cluster_view.get_selected_ids() == [10, 30]
    assert supervisor.similarity_view.get_selected_ids() == [20, 11]


def test_supervisor_merge_view_opens_below_cluster_and_restores_position(qtbot, supervisor):
    _select(supervisor, [30], [20])

    supervisor.toggle_merge_mode()
    qtbot.wait(10)
    merge_view = supervisor.merge_view
    merge_dock = merge_view.dock
    cluster_rect = supervisor.cluster_view.dock.geometry()
    merge_rect = merge_dock.geometry()
    assert merge_rect.top() >= cluster_rect.bottom()

    supervisor.merge_view.dock.setFloating(True)
    supervisor.merge_view.dock.move(70, 80)
    supervisor.merge_view.dock.resize(240, 180)
    qtbot.wait(10)
    floating_position = supervisor.merge_view.dock.pos()
    floating_size = supervisor.merge_view.dock.size()

    supervisor.toggle_merge_mode()
    supervisor.toggle_merge_mode()
    qtbot.wait(10)

    assert supervisor.merge_view is merge_view
    assert supervisor.merge_view.dock is merge_dock
    assert supervisor.merge_view.dock.isFloating()
    assert supervisor.merge_view.dock.pos() == floating_position
    assert supervisor.merge_view.dock.size() == floating_size


def test_supervisor_merge_view_restores_docked_extent(qtbot, supervisor):
    _select(supervisor, [30], [20])
    supervisor.toggle_merge_mode()
    dock = supervisor.merge_view.dock
    supervisor.gui.resizeDocks((dock,), (210,), Qt.Horizontal)
    supervisor.gui.resizeDocks((dock,), (160,), Qt.Vertical)
    qtbot.wait(10)
    docked_size = dock.size()

    supervisor.toggle_merge_mode()
    supervisor.toggle_merge_mode()
    qtbot.wait(10)

    assert not dock.isFloating()
    assert dock.size() == docked_size


def test_merge_view_header_has_no_sort_affordance(supervisor):
    _select(supervisor, [30], [20])
    supervisor.toggle_merge_mode()
    view = supervisor.merge_view
    header = view.table_view.horizontalHeader()
    ids = view.get_ids()

    assert not header.isSortIndicatorShown()
    assert not header.sectionsClickable()
    assert header.sectionsMovable()
    view._on_header_clicked(0)

    assert view.get_ids() == ids
    assert view._current_sort is None


def test_supervisor_merge_candidate_interactions_follow_visible_role_order(supervisor):
    _select(supervisor, [10, 30], [20])
    supervisor.toggle_merge_mode()
    candidate = supervisor.similarity_view.get_ids()[0]
    events = []

    @connect(sender=supervisor)
    def on_select(sender, cluster_ids):
        events.append(cluster_ids)

    supervisor.similarity_view.select([candidate])
    supervisor.block()
    assert events == [[10, 30, 20, candidate]]
    colors_before = {
        cluster_id: supervisor.merge_view._selected_color_index(cluster_id)
        for cluster_id in supervisor.selected
    }
    events.clear()

    supervisor.add_to_merge((candidate,))
    assert supervisor.selected_merge == [10, 30, 20, candidate]
    assert supervisor.selected_similar == []
    assert events == []

    supervisor.remove_from_merge(30)
    assert supervisor.selected_merge == [10, 20, candidate]
    assert supervisor.selected_similar == [30]
    assert supervisor.selected == [10, 20, candidate, 30]
    assert events == [[10, 20, candidate, 30]]
    assert supervisor.similarity_view._selected_color_index(30) == colors_before[30]
    events.clear()

    supervisor.reorder_merge((candidate,), 1)
    assert supervisor.selected_merge == [10, candidate, 20]
    assert supervisor.selected == [10, candidate, 20, 30]
    assert events == [[10, candidate, 20, 30]]
    assert {
        cluster_id: (
            supervisor.merge_view._selected_color_index(cluster_id)
            if cluster_id in supervisor.selected_merge
            else supervisor.similarity_view._selected_color_index(cluster_id)
        )
        for cluster_id in supervisor.selected
    } == colors_before
    unconnect(on_select)


def test_merge_backspace_removes_similarity_tail_without_recoloring_merge(supervisor):
    _select(supervisor, [10, 30], [20])
    supervisor.toggle_merge_mode()
    candidate = supervisor.similarity_view.get_ids()[0]
    supervisor.similarity_view.select([candidate])
    supervisor.block()
    colors_before = {
        cluster_id: supervisor.merge_view._selected_color_index(cluster_id)
        for cluster_id in supervisor.selected_merge
    }

    supervisor.select_actions.unselect_similar()
    supervisor.block()

    assert supervisor.selected == supervisor.selected_merge
    assert supervisor.selected_similar == []
    assert {
        cluster_id: supervisor.merge_view._selected_color_index(cluster_id)
        for cluster_id in supervisor.selected_merge
    } == colors_before


def test_supervisor_merge_drag_drop_intents(supervisor):
    _select(supervisor, [10, 30], [20])
    supervisor.toggle_merge_mode()
    candidate = supervisor.similarity_view.get_ids()[0]
    assert supervisor.merge_view.table_view.acceptDrops()
    assert supervisor.similarity_view.table_view.acceptDrops()
    assert not supervisor.cluster_view.table_view.acceptDrops()
    assert supervisor.merge_view.table_view.selectionMode() == QAbstractItemView.SingleSelection
    movable = supervisor.merge_view._proxy_index_for_id(30)
    supervisor.merge_view.table_view.setCurrentIndex(movable)
    assert supervisor.merge_view._drag_ids_for_index(movable) == (30,)
    reference = supervisor.merge_view._proxy_index_for_id(10)
    assert supervisor.merge_view._drag_ids_for_index(reference) == (10,)

    supervisor.merge_view.emit_cluster_drop(supervisor.similarity_view, (candidate,), 1)
    assert supervisor.selected_merge == [10, candidate, 30, 20]

    supervisor.merge_view.emit_cluster_drop(supervisor.merge_view, (20,), 1)
    assert supervisor.selected_merge == [10, 20, candidate, 30]

    supervisor.similarity_view.emit_cluster_drop(supervisor.merge_view, (candidate,), 0)
    assert supervisor.selected_merge == [10, 20, 30]
    assert supervisor.selected_similar == [candidate]

    supervisor.toggle_merge_mode()
    assert not supervisor.similarity_view.table_view.dragEnabled()


def test_supervisor_plain_right_click_transfers_in_both_modes(qtbot, supervisor):
    _select(supervisor, [10, 30], [20])
    candidate = next(
        cluster_id
        for cluster_id in supervisor.similarity_view.get_ids()
        if cluster_id not in supervisor.selected_similar
    )
    index = supervisor.similarity_view._proxy_index_for_id(candidate)
    pos = supervisor.similarity_view.table_view.visualRect(index).center()
    qtbot.mouseClick(
        supervisor.similarity_view.table_view.viewport(),
        Qt.RightButton,
        pos=pos,
    )
    supervisor.block()
    assert candidate in supervisor.selected_clusters
    assert candidate not in supervisor.selected_similar

    cluster_candidate = next(
        cluster_id
        for cluster_id in supervisor.cluster_view.get_ids()
        if cluster_id not in supervisor.selected
    )
    index = supervisor.cluster_view._proxy_index_for_id(cluster_candidate)
    pos = supervisor.cluster_view.table_view.visualRect(index).center()
    qtbot.mouseClick(supervisor.cluster_view.table_view.viewport(), Qt.RightButton, pos=pos)
    supervisor.block()
    assert cluster_candidate in supervisor.selected_similar

    supervisor.toggle_merge_mode()
    next_candidate = next(
        cluster_id
        for cluster_id in supervisor.similarity_view.get_ids()
        if cluster_id not in supervisor.selected_similar
    )
    index = supervisor.similarity_view._proxy_index_for_id(next_candidate)
    pos = supervisor.similarity_view.table_view.visualRect(index).center()

    qtbot.mouseClick(
        supervisor.similarity_view.table_view.viewport(),
        Qt.RightButton,
        pos=pos,
    )
    supervisor.block()
    assert next_candidate in supervisor.selected_merge

    supervisor._transfer_row_on_right_click(supervisor.merge_view, next_candidate)
    supervisor.block()
    assert next_candidate not in supervisor.selected_merge
    assert next_candidate in supervisor.selected_similar


def test_supervisor_reference_transfer_promotes_and_recomputes(qtbot, supervisor):
    _select(supervisor, [10, 30], [20])
    supervisor.toggle_merge_mode()

    index = supervisor.merge_view._proxy_index_for_id(10)
    pos = supervisor.merge_view.table_view.visualRect(index).center()
    qtbot.mouseClick(supervisor.merge_view.table_view.viewport(), Qt.RightButton, pos=pos)
    supervisor.block()

    assert supervisor.selected_merge == [30, 20]
    assert supervisor.selected_similar == [10]
    assert supervisor.selection.state.reference_id == 30
    assert supervisor.merge_view._selected_color_index(30) == 0
    assert 10 in supervisor.similarity_view.get_ids()


def test_closing_merge_view_cancels_mode(supervisor):
    _select(supervisor, [30], [20])
    entry = supervisor.selection.snapshot()
    supervisor.toggle_merge_mode()
    merge_view = supervisor.merge_view

    merge_view.dock.close()

    assert supervisor.selection.state == entry
    assert supervisor.merge_view is merge_view
    assert merge_view in supervisor.gui.views
    assert merge_view.dock.isHidden()
    assert supervisor.cluster_view.isEnabled()


def test_merge_mode_action_and_cancel_control(supervisor):
    _select(supervisor, [30], [20])

    supervisor.select_actions.toggle_merge_mode()
    supervisor.block()
    assert supervisor.selection.state.is_merge_mode
    merge_view = supervisor.merge_view

    supervisor.merge_view.dock.get_widget('cancel_merge_mode').click()
    supervisor.block()
    assert not supervisor.selection.state.is_merge_mode
    assert supervisor.merge_view is merge_view
    assert merge_view.dock.isHidden()


def test_merge_mode_rejects_cluster_mutations(supervisor):
    _select(supervisor, [30], [20])
    supervisor.toggle_merge_mode()
    state = supervisor.selection.state

    supervisor.select([10])
    supervisor.split([0, 1])
    supervisor.label('group', 'noise', [30])
    supervisor.sort('id')
    supervisor.filter('id > 0')
    supervisor.first()
    supervisor.next_best()
    supervisor.merge([30, 20, 10])

    assert supervisor.selection.state is state
    assert set(supervisor.clustering.cluster_ids) >= {30, 20}


def test_restore_parents_preserves_later_edits_and_workspace_history(supervisor, qtbot):
    _select(supervisor, [30], [20])
    groups = {p: supervisor.clustering.spikes_per_cluster[p].copy() for p in (30, 20)}
    merged = supervisor.merge().added[0]
    supervisor.block()
    _select(supervisor, [10], [11])
    unrelated = supervisor.merge().added[0]
    supervisor.block()
    supervisor.label('group', 'mua', [unrelated])
    supervisor.block()
    supervisor.label('group', 'good', [merged])
    supervisor.block()
    _select(supervisor, [merged])
    before = supervisor.clustering.spike_clusters.copy()
    position = supervisor._global_history.current_position
    out = supervisor.restore_parents(merged, groups=groups, colors={30: 0, 20: 1})
    supervisor.block()
    children = list(out.parent_restore_origins)
    assert supervisor.selected_merge == children
    assert supervisor.actions.get('undo').isEnabled()
    assert supervisor._global_history.current_position == position + 1
    assert supervisor.cluster_meta.get('group', unrelated) == 'mua'
    assert all(supervisor.cluster_meta.get('group', child) is None for child in children)
    assert supervisor.get_cluster_info(children[0])['parent_origin'] == 'was 30'
    assert supervisor.merge_view.column_order()[:2] == ['id', 'parent_origin']
    parent_column = supervisor.merge_view.columns.index('parent_origin')
    assert supervisor.merge_view._model.headerData(parent_column, Qt.Horizontal) == 'Parent'
    assert supervisor.cluster_view._model.headerData(0, Qt.Horizontal) == 'id'
    restored = supervisor.clustering.spike_clusters.copy()

    supervisor.undo()
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, before)
    assert supervisor.selected == [merged]
    assert supervisor.cluster_meta.get('group', merged) == 'good'
    assert supervisor.cluster_meta.get('group', unrelated) == 'mua'
    supervisor.redo()
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, restored)
    assert supervisor.selected_merge == children
    assert all(supervisor.cluster_meta.get('group', child) is None for child in children)

    # The visible exclusion button acts on an actual clicked row. G must not
    # include that parent through an accidental Similarity selection.
    view = supervisor.merge_view
    index = view._proxy_index_for_id(children[1])
    qtbot.mouseClick(
        view.table_view.viewport(), Qt.LeftButton, pos=view.table_view.visualRect(index).center()
    )
    qtbot.mouseClick(view.dock.get_widget('exclude_merge_row'), Qt.LeftButton)
    supervisor.block()
    assert children[1] not in supervisor.selected
    assert children[1] in supervisor.clustering.cluster_ids
    supervisor.toggle_merge_mode()
    supervisor.block()
    assert supervisor.selected_clusters == children
    ae(supervisor.clustering.spike_clusters, restored)

    supervisor.toggle_merge_mode()
    supervisor.deselect_from_merge([children[1]])
    supervisor.add_to_merge([2])
    supervisor.action_creator.edit_actions.get('merge').trigger()
    supervisor.block()
    assert children[1] in supervisor.clustering.cluster_ids
    result = supervisor.selected_clusters[0]
    ae(
        supervisor.clustering.spikes_per_cluster[result],
        np.sort(np.concatenate([groups[30], [4, 5]])),
    )
    assert supervisor.cluster_meta.get('group', unrelated) == 'mua'


def test_restore_workspace_physical_right_click_preserves_transfer_contract(supervisor, qtbot):
    _select(supervisor, [30], [20])
    groups = {p: supervisor.clustering.spikes_per_cluster[p].copy() for p in (30, 20)}
    target = supervisor.merge().added[0]
    supervisor.block()
    out = supervisor.restore_parents(target, groups=groups)
    supervisor.block()
    child = list(out.parent_restore_origins)[1]
    view = supervisor.merge_view
    index = view._proxy_index_for_id(child)
    qtbot.mouseClick(
        view.table_view.viewport(), Qt.RightButton, pos=view.table_view.visualRect(index).center()
    )
    supervisor.block()
    assert child not in supervisor.selected_merge
    assert child in supervisor.selected_similar
    # Transfer deliberately preserves the effective selection; exclusion has
    # its own explicit button and is tested above.
    assert child in supervisor.selected


@mark.parametrize('restored_quality', [None, 'mua'])
def test_remerge_restored_parent_undo_preserves_exact_metadata(supervisor, restored_quality):
    _select(supervisor, [30], [20])
    groups = {p: supervisor.clustering.spikes_per_cluster[p].copy() for p in (30, 20)}
    target = supervisor.merge().added[0]
    supervisor.block()
    out = supervisor.restore_parents(target, groups=groups)
    supervisor.block()
    first, excluded = out.parent_restore_origins
    supervisor.toggle_merge_mode()
    if restored_quality is not None:
        supervisor.label('group', restored_quality, [first])
        supervisor.block()
    assert supervisor.cluster_meta.get('group', 1) == 'good'
    _select(supervisor, [first], [1])
    supervisor.toggle_merge_mode()
    supervisor.merge()
    supervisor.block()
    supervisor.undo()
    supervisor.block()
    assert supervisor.cluster_meta.get('group', first) == restored_quality
    assert supervisor.cluster_meta.get('group', excluded) is None
    assert supervisor.cluster_meta.get('group', 1) == 'good'
    assert first in supervisor.selected_merge
    supervisor.redo()
    supervisor.block()
    assert excluded in supervisor.clustering.cluster_ids


def test_restore_parents_rolls_back_observer_failure(supervisor, monkeypatch):
    _select(supervisor, [30], [20])
    groups = {p: supervisor.clustering.spikes_per_cluster[p].copy() for p in (30, 20)}
    target = supervisor.merge().added[0]
    supervisor.block()
    before = supervisor.clustering.spike_clusters.copy()
    selection = supervisor.selection.snapshot()
    history = supervisor._global_history.current_position
    workflow = supervisor._workflow_context()

    def fail(*args, **kwargs):
        raise RuntimeError('injected restore presentation failure')

    monkeypatch.setattr(supervisor, '_show_merge_view', fail)
    with raises(RuntimeError, match='injected'):
        supervisor.restore_parents(target, groups=groups)
    ae(supervisor.clustering.spike_clusters, before)
    assert supervisor.selection.state == selection
    assert supervisor._global_history.current_position == history
    assert supervisor._workflow_context() == workflow


def test_merge_mode_next_navigates_similarity_not_cluster(supervisor):
    _select(supervisor, [30], [20])
    supervisor.toggle_merge_mode()
    before = supervisor.selection.state

    supervisor.next()
    supervisor.block()

    assert supervisor.selection.state.is_merge_mode
    assert supervisor.selection.state.merge is before.merge
    assert supervisor.selected_clusters == []
    assert len(supervisor.selected_similar) == 1
    first_candidate = supervisor.selected_similar[0]
    first_colors = dict(supervisor.selection_color_indices)

    supervisor.next()
    supervisor.block()

    assert {
        cluster_id: supervisor.selection_color_indices[cluster_id] for cluster_id in first_colors
    } == first_colors
    assert supervisor.selected_similar != [first_candidate]


def test_merge_mode_merge_undo_redo_restores_workspace(supervisor):
    _select(supervisor, [30], [20])
    assignments_before = supervisor.clustering.spike_clusters.copy()
    supervisor.toggle_merge_mode()
    candidate = supervisor.similarity_view.get_ids()[0]
    supervisor.similarity_view.select([candidate])
    supervisor.block()
    merge_before = supervisor.selection.snapshot()
    colors_before = {
        cluster_id: (
            supervisor.merge_view._selected_color_index(cluster_id)
            if cluster_id in supervisor.selected_merge
            else supervisor.similarity_view._selected_color_index(cluster_id)
        )
        for cluster_id in supervisor.selected
    }

    up = supervisor.merge()
    supervisor.block()

    merged_id = up.added[0]
    assert not supervisor.selection.state.is_merge_mode
    assert supervisor.selected_merge == []
    assert supervisor.selected_clusters == [merged_id]
    assert supervisor.selected_similar == []
    assert supervisor.selected == [merged_id]
    merge_view = supervisor.merge_view
    assert merge_view is not None
    assert merge_view.dock.isHidden()
    assert supervisor.actions.get('undo').isEnabled()
    assert set(up.deleted) == {30, 20, candidate}
    assignments_after = supervisor.clustering.spike_clusters.copy()
    events = []

    @connect(sender=supervisor)
    def on_select(sender, cluster_ids):
        events.append(cluster_ids)

    supervisor.undo()
    supervisor.block()

    ae(supervisor.clustering.spike_clusters, assignments_before)
    assert supervisor.selection.state == merge_before
    assert supervisor.selected_merge == [30, 20]
    assert supervisor.selected_similar == [candidate]
    assert supervisor.merge_view is merge_view
    assert not merge_view.dock.isHidden()
    assert merge_view.dock.get_widget('cancel_merge_mode').text() == 'Cancel Merge Mode'
    assert supervisor.actions.get('redo').isEnabled()
    assert events[-1] == list(merge_before.presentation_order)
    assert dict(supervisor.selection_color_indices) == dict(merge_before.color_indices)
    assert {
        cluster_id: (
            supervisor.merge_view._selected_color_index(cluster_id)
            if cluster_id in supervisor.selected_merge
            else supervisor.similarity_view._selected_color_index(cluster_id)
        )
        for cluster_id in supervisor.selected
    } == colors_before

    supervisor.redo()
    supervisor.block()

    ae(supervisor.clustering.spike_clusters, assignments_after)
    assert not supervisor.selection.state.is_merge_mode
    assert supervisor.selected_merge == []
    assert supervisor.selected_clusters == [merged_id]
    assert supervisor.selected == [merged_id]
    assert supervisor.merge_view is merge_view
    assert merge_view.dock.isHidden()
    assert supervisor.actions.get('undo').isEnabled()
    assert events[-1] == [merged_id]
    unconnect(on_select)


def test_merge_mode_merge_includes_filtered_selected_similarity(supervisor):
    _select(supervisor, [30], [20])
    supervisor.toggle_merge_mode()
    candidate = supervisor.similarity_view.get_ids()[0]
    supervisor.similarity_view.select([candidate])
    supervisor.block()
    supervisor.similarity_view.filter(f'id != {candidate}')

    assert supervisor.selected_similar == [candidate]
    assert candidate not in supervisor.similarity_view.get_selected_ids()

    up = supervisor.merge()
    supervisor.block()

    assert set(up.deleted) == {30, 20, candidate}


def test_manual_merge_returns_to_visible_cluster_for_quality_assignment(monkeypatch, supervisor):
    _select(supervisor, [30], [20])
    supervisor.cluster_view.sort_by('id', 'asc')
    supervisor.cluster_view.filter('id >= 0')
    scrolled = []
    monkeypatch.setattr(
        supervisor.cluster_view,
        'scroll_to',
        lambda cluster_id, center: scrolled.append((cluster_id, center)),
    )
    supervisor.toggle_merge_mode()
    merged_id = supervisor.merge().added[0]
    supervisor.block()

    assert supervisor.cluster_view.filter_edit.text() == ''
    assert supervisor.cluster_view._current_sort == ('id', 'asc')
    assert scrolled == [(merged_id, True)]
    supervisor.move('good', 'all')

    assert not supervisor.selection.state.is_merge_mode
    assert supervisor.selected == [merged_id]
    assert supervisor.cluster_meta.get('group', merged_id) == 'good'


def test_manual_merge_supports_explicit_chained_merge_and_direct_undo(supervisor):
    _select(supervisor, [30], [20])
    supervisor.toggle_merge_mode()
    first_id = supervisor.merge().added[0]
    supervisor.block()
    candidate = supervisor.similarity_view.get_ids()[0]
    supervisor.toggle_merge_mode()
    supervisor.similarity_view.select([candidate])
    supervisor.block()
    before_second = supervisor.selection.snapshot()

    second = supervisor.merge()
    supervisor.block()

    second_id = second.added[0]
    assert set(second.deleted) == {first_id, candidate}
    assert supervisor.selected_clusters == [second_id]
    assert supervisor.selected_merge == []

    supervisor.undo()
    supervisor.block()

    assert supervisor.selection.state == before_second
    assert supervisor.selected_merge == [first_id]
    assert supervisor.selected_similar == [candidate]


def test_manual_merge_supports_action_dragged_explicit_chained_merge(supervisor):
    _select(supervisor, [30], [20])
    supervisor.toggle_merge_mode()
    supervisor.action_creator.edit_actions.get('merge').trigger()
    supervisor.block()
    first_id = supervisor.selected_clusters[0]
    assert not supervisor.selection.state.is_merge_mode
    supervisor.toggle_merge_mode()
    candidate = supervisor.similarity_view.get_ids()[0]

    supervisor.merge_view.emit_cluster_drop(supervisor.similarity_view, (candidate,), 1)

    assert supervisor.selected_merge == [first_id, candidate]
    assert supervisor.selected_similar == []

    supervisor.action_creator.edit_actions.get('merge').trigger()
    supervisor.block()

    assert first_id not in supervisor.clustering.cluster_ids
    assert candidate not in supervisor.clustering.cluster_ids
    assert len(supervisor.selected_clusters) == 1
    assert supervisor.selected_merge == []


def test_uncommitted_merge_workspace_does_not_undo_prior_action(supervisor):
    _select(supervisor, [30], [20])
    supervisor.merge()
    supervisor.block()
    merged_selection = supervisor.selection.snapshot()
    supervisor.toggle_merge_mode()

    supervisor.undo()

    assert supervisor.selection.state.is_merge_mode
    assert supervisor.selection.state.merge.entry_snapshot.selection == merged_selection


def test_failed_merge_preserves_complete_merge_workspace(monkeypatch, supervisor):
    _select(supervisor, [30], [20])
    supervisor.toggle_merge_mode()
    state = supervisor.selection.state
    rows = supervisor.merge_view.get_ids()

    def fail(*args, **kwargs):
        raise RuntimeError('merge failed')

    monkeypatch.setattr(supervisor.clustering, 'merge', fail)
    with raises(RuntimeError, match='merge failed'):
        supervisor.merge()

    assert supervisor.selection.state is state
    assert supervisor.merge_view.get_ids() == rows
    assert supervisor.cluster_view._interaction_blocked


def _proposition_supervisor(
    gui,
    cluster_ids,
    cluster_groups,
    cluster_labels,
    similarity,
    tempdir,
    merges=None,
):
    merges = merges or [
        {'unit_ids': [30, 20]},
        {'unit_ids': [20, 10]},
        {'unit_ids': [11, 1]},
    ]
    catalog = decode_curation_mapping(
        {
            'format_version': '2',
            'unit_ids': cluster_ids,
            'merges': merges,
        }
    )
    supervisor = Supervisor(
        np.repeat(cluster_ids, 2),
        cluster_groups=cluster_groups,
        cluster_labels=cluster_labels,
        similarity=similarity,
        context=Context(tempdir),
        merge_propositions=MergePropositionController(catalog),
    )
    supervisor.attach(gui)
    barrier = Barrier()
    connect(barrier('cluster_view'), event='ready', sender=supervisor.cluster_view)
    connect(barrier('similarity_view'), event='ready', sender=supervisor.similarity_view)
    barrier.wait()
    return supervisor


def test_ordinary_restoration_preserves_accepted_review_with_derived_status(
    gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    from pathlib import Path
    from types import SimpleNamespace

    from phy.apps._parent_review import ParentInspection, ParentLineage

    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    lineage = ParentLineage(
        SimpleNamespace(
            dir_path=Path(tempdir),
            supervisor=supervisor,
            model=SimpleNamespace(spike_times=np.arange(supervisor.clustering.n_spikes)),
        )
    )
    supervisor.parent_lineage = lineage
    try:
        key = supervisor.merge_propositions.catalog.propositions[0].key
        supervisor._activate_merge_proposition(supervisor.merge_propositions_view, key)
        target = supervisor.merge().added[0]
        supervisor.block()
        review = supervisor.merge_propositions.catalog.reviews[key]
        assert review.result_unit_id == target
        supervisor.restore_parents(target)
        supervisor.block()
        assert supervisor.merge_propositions.catalog.reviews[key] == review
        assert supervisor.merge_propositions.catalog.status_for(key).value == 'accepted_restored'
        emit('curation_saved', supervisor)
        reopened = Supervisor(
            supervisor.clustering.spike_clusters.copy(),
            merge_propositions=MergePropositionController(supervisor.merge_propositions.catalog),
        )
        # Producer/review loading does not supply ordinary retirement projection;
        # the saved parent sidecar must establish it before GUI attachment.
        reopened.merge_propositions.project_live_ids(
            reopened.clustering.cluster_ids, retired_units={}
        )
        assert reopened.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
        inspection = ParentInspection(
            SimpleNamespace(
                dir_path=Path(tempdir),
                supervisor=reopened,
                model=SimpleNamespace(spike_times=np.arange(reopened.clustering.n_spikes)),
            )
        )
        try:
            assert reopened.merge_propositions.catalog.status_for(key).value == 'accepted_restored'
        finally:
            inspection.close()
        supervisor.undo()
        supervisor.block()
        assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
        supervisor.redo()
        supervisor.block()
        assert supervisor.merge_propositions.catalog.status_for(key).value == 'accepted_restored'
        supervisor.toggle_merge_mode()
    finally:
        lineage.close()


@fixture
def lineage_supervisor(gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir):
    from pathlib import Path
    from types import SimpleNamespace

    from phy.apps._parent_review import ParentLineage

    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    lineage = ParentLineage(
        SimpleNamespace(
            dir_path=Path(tempdir),
            supervisor=supervisor,
            model=SimpleNamespace(spike_times=np.arange(supervisor.clustering.n_spikes)),
        )
    )
    supervisor.parent_lineage = lineage
    try:
        yield supervisor
    finally:
        lineage.close()


def _accept_first_proposition(supervisor):
    key = supervisor.merge_propositions.catalog.propositions[0].key
    supervisor._activate_merge_proposition(supervisor.merge_propositions_view, key)
    target = supervisor.merge().added[0]
    supervisor.block()
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
    return key, target


def test_recursive_restore_reaches_original_components_as_one_undoable_action(
    lineage_supervisor,
):
    supervisor = lineage_supervisor
    original = supervisor.clustering.spike_clusters.copy()
    key, first = _accept_first_proposition(supervisor)
    _select(supervisor, [first], [10])
    nested = supervisor.merge().added[0]
    supervisor.block()
    _select(supervisor, [0], [2])
    unrelated = supervisor.merge().added[0]
    supervisor.block()
    supervisor.label('group', 'mua', [unrelated])
    supervisor.block()
    _select(supervisor, [nested])
    before = supervisor.clustering.spike_clusters.copy()
    history_position = supervisor._global_history.current_position
    review = supervisor.merge_propositions.catalog.reviews[key]

    out = supervisor.restore_parents(nested)
    supervisor.block()

    assert set(out.parent_restore_origins.values()) == {30, 20, 10}
    assert len(out.added) == 3
    assert set(out.added).isdisjoint({30, 20, 10, first, nested})
    for child, parent in out.parent_restore_origins.items():
        ae(supervisor.clustering.spikes_per_cluster[child], np.flatnonzero(original == parent))
        assert supervisor.get_cluster_info(child)['parent_origin'] == f'was {parent}'
    assert supervisor._global_history.current_position == history_position + 1
    assert supervisor.cluster_meta.get('group', unrelated) == 'mua'
    assert (
        supervisor.merge_propositions.catalog.status_for(key)
        is PropositionStatus.ACCEPTED_RESTORED
    )
    assert supervisor.merge_propositions.catalog.reviews[key] == review
    restored = supervisor.clustering.spike_clusters.copy()

    supervisor.undo()
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, before)
    assert supervisor.cluster_meta.get('group', unrelated) == 'mua'
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
    supervisor.redo()
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, restored)
    assert supervisor.selected_merge == list(out.parent_restore_origins)
    assert supervisor.cluster_meta.get('group', unrelated) == 'mua'
    supervisor.toggle_merge_mode()
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, restored)


def test_ordinary_exact_remerge_recovers_green_through_undo_redo_and_repeated_restore(
    lineage_supervisor,
):
    supervisor = lineage_supervisor
    key, target = _accept_first_proposition(supervisor)
    historical_review = supervisor.merge_propositions.catalog.review_mapping()
    for _ in range(2):
        out = supervisor.restore_parents(target)
        supervisor.block()
        assert set(out.parent_restore_origins.values()) == {30, 20}
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_RESTORED
        )
        restored = supervisor.clustering.spike_clusters.copy()
        # Exercise the real G action from the editable workspace.
        supervisor.action_creator.edit_actions.get('merge').trigger()
        supervisor.block()
        target = supervisor.selected_clusters[0]
        remerged = supervisor.clustering.spike_clusters.copy()
        assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
        assert supervisor.merge_propositions_view._model.row_by_id(0)['status'] == 'accepted'
        assert supervisor.merge_propositions.catalog.review_mapping() == historical_review
        supervisor.undo()
        supervisor.block()
        ae(supervisor.clustering.spike_clusters, restored)
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_RESTORED
        )
        supervisor.redo()
        supervisor.block()
        ae(supervisor.clustering.spike_clusters, remerged)
        assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
        assert supervisor.merge_propositions.catalog.review_mapping() == historical_review


def test_failed_restore_discards_temporary_reassembly_cache(lineage_supervisor, monkeypatch):
    supervisor = lineage_supervisor
    key, target = _accept_first_proposition(supervisor)
    supervisor.restore_parents(target)
    supervisor.action_creator.edit_actions.get('merge').trigger()
    supervisor.block()
    target = supervisor.selected_clusters[0]
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
    assignments = supervisor.clustering.spike_clusters.copy()
    selection = supervisor.selection.snapshot()
    history = supervisor._global_history.current_position
    reviews = supervisor.merge_propositions.catalog.review_mapping()
    restore = supervisor._restore_parent_partition

    def fail_after_refresh(*args, **kwargs):
        restore(*args, **kwargs)
        assert (
            supervisor.merge_propositions.catalog.status_for(key)
            is PropositionStatus.ACCEPTED_RESTORED
        )
        assert supervisor.parent_lineage._reassembly_cache[1] == {}
        raise RuntimeError('injected failure after restoration refresh')

    monkeypatch.setattr(supervisor, '_restore_parent_partition', fail_after_refresh)
    with raises(RuntimeError, match='after restoration refresh'):
        supervisor.restore_parents(target)
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, assignments)
    assert supervisor.selection.state == selection
    assert supervisor._global_history.current_position == history
    assert supervisor.merge_propositions.catalog.review_mapping() == reviews
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
    row_id = supervisor.merge_propositions_view._id_by_key[key]
    row = supervisor.merge_propositions_view._model.row_by_id(row_id)
    assert row['status'] == 'accepted'
    assert (
        supervisor.merge_propositions_view._foreground_color(row, 'proposition').name()
        == '#86d16d'
    )


def test_partial_remerge_stays_white_then_complete_with_extra_becomes_modified_green(
    lineage_supervisor,
):
    supervisor = lineage_supervisor
    key, target = _accept_first_proposition(supervisor)
    historical_review = supervisor.merge_propositions.catalog.review_mapping()
    out = supervisor.restore_parents(target)
    supervisor.block()
    _, excluded = out.parent_restore_origins
    supervisor.deselect_from_merge([excluded])
    supervisor.add_to_merge([10])
    supervisor.action_creator.edit_actions.get('merge').trigger()
    supervisor.block()
    partial = supervisor.selected_clusters[0]
    assert excluded in supervisor.clustering.cluster_ids
    assert (
        supervisor.merge_propositions.catalog.status_for(key)
        is PropositionStatus.ACCEPTED_RESTORED
    )

    _select(supervisor, [partial, excluded])
    supervisor.toggle_merge_mode()
    supervisor.action_creator.edit_actions.get('merge').trigger()
    supervisor.block()
    assert (
        supervisor.merge_propositions.catalog.status_for(key)
        is PropositionStatus.ACCEPTED_MODIFIED
    )
    assert supervisor.merge_propositions.catalog.review_mapping() == historical_review
    supervisor.undo()
    supervisor.block()
    assert (
        supervisor.merge_propositions.catalog.status_for(key)
        is PropositionStatus.ACCEPTED_RESTORED
    )
    supervisor.redo()
    supervisor.block()
    assert (
        supervisor.merge_propositions.catalog.status_for(key)
        is PropositionStatus.ACCEPTED_MODIFIED
    )
    assert supervisor.merge_propositions.catalog.review_mapping() == historical_review


def test_split_membership_revokes_reassembled_acceptance_even_with_origin_aliases(
    lineage_supervisor,
):
    supervisor = lineage_supervisor
    key, target = _accept_first_proposition(supervisor)
    out = supervisor.restore_parents(target)
    supervisor.block()
    source_child = next(
        child for child, parent in out.parent_restore_origins.items() if parent == 30
    )
    one_spike = supervisor.clustering.spikes_per_cluster[source_child][:1].copy()
    supervisor.action_creator.edit_actions.get('merge').trigger()
    supervisor.block()
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
    before_split = supervisor.clustering.spike_clusters.copy()
    supervisor.split(one_spike)
    supervisor.block()
    assert (
        supervisor.merge_propositions.catalog.status_for(key)
        is PropositionStatus.ACCEPTED_RESTORED
    )
    supervisor.undo()
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, before_split)
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.ACCEPTED
    supervisor.redo()
    supervisor.block()
    assert (
        supervisor.merge_propositions.catalog.status_for(key)
        is PropositionStatus.ACCEPTED_RESTORED
    )


def test_merge_proposition_review_cancel_restores_exact_entry(
    gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    _select(supervisor, [11], [10])
    entry = supervisor.selection.snapshot()
    key = supervisor.merge_propositions.catalog.propositions[0].key
    view = supervisor.merge_propositions_view

    assert view.columns == ['proposition']
    assert not hasattr(view, 'action_buttons')
    assert [view._model.row_by_id(i)['display_id'] for i in range(3)] == ['P1', 'P2', 'P3']
    assert view.select_key(key)
    assert supervisor.selection.snapshot() is entry
    supervisor.toggle_merge_mode()
    assert supervisor.selection.state.is_merge_mode
    assert supervisor.selection.state.merge.proposition_id is None
    merge_view = supervisor.merge_view
    merge_dock = merge_view.dock
    events = []

    @connect(sender=supervisor)
    def on_select(sender, cluster_ids):
        events.append(cluster_ids)

    view._on_row_clicked(view._proxy_index_for_id(view._id_by_key[key]))

    assert supervisor.merge_view is merge_view
    assert supervisor.merge_view.dock is merge_dock
    assert not merge_dock.isHidden()
    assert events == [[30, 20]]
    assert supervisor.selected_merge == [30, 20]
    assert supervisor.selection.state.reference_id == 30
    assert supervisor.selection.state.color_indices[30] == 0
    assert supervisor.selection.state.merge.proposition_id == key
    assert 'PROPOSITION merge:' in supervisor.merge_view.dock.status

    replacement = supervisor.merge_propositions.catalog.propositions[2]
    cluster_geometry = supervisor.cluster_view.dock.geometry()
    proposition_geometry = supervisor.merge_propositions_view.dock.geometry()
    view._on_row_clicked(view._proxy_index_for_id(view._id_by_key[replacement.key]))
    assert supervisor.merge_view is merge_view
    assert supervisor.merge_view.dock is merge_dock
    assert not merge_dock.isHidden()
    assert events == [[30, 20], [11, 1]]
    assert supervisor.cluster_view.dock.geometry() == cluster_geometry
    assert supervisor.merge_propositions_view.dock.geometry() == proposition_geometry
    assert supervisor.selected_merge == [11, 1]
    assert supervisor.selection.state.merge.proposition_id == replacement.key

    supervisor.toggle_merge_mode()
    assert supervisor.selection.state == entry
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.PENDING
    unconnect(on_select)


def test_merge_proposition_navigation_shortcuts_and_text_focus(
    gui, qtbot, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    first, second, third = supervisor.merge_propositions.catalog.propositions
    shortcuts = {
        name: _get_shortcut_string(supervisor.select_actions.get(name).shortcut())
        for name in (
            'next_merge_proposition',
            'previous_merge_proposition',
            'reject_merge_proposition',
            'reset_merge_proposition',
        )
    }
    assert shortcuts == {
        'next_merge_proposition': 'alt+down',
        'previous_merge_proposition': 'alt+up',
        'reject_merge_proposition': 'alt+backspace',
        'reset_merge_proposition': 'alt+shift+backspace',
    }

    supervisor._activate_merge_proposition(supervisor.merge_propositions_view, first.key)
    merge_view = supervisor.merge_view
    merge_dock = merge_view.dock
    supervisor.next_merge_proposition()
    assert supervisor.merge_view is merge_view
    assert supervisor.merge_view.dock is merge_dock
    assert not merge_dock.isHidden()
    assert supervisor.selection.state.merge.proposition_id == second.key
    assert supervisor.merge_propositions_view._model.row_by_id(0)['status'] == 'pending'
    assert supervisor.merge_propositions_view._model.row_by_id(1)['status'] == 'active'
    supervisor.previous_merge_proposition()
    assert supervisor.merge_view is merge_view
    assert supervisor.merge_view.dock is merge_dock
    assert supervisor.selection.state.merge.proposition_id == first.key

    supervisor.merge_propositions_view.filter_edit.setFocus()
    qtbot.wait(1)
    supervisor.next_merge_proposition()
    assert supervisor.selection.state.merge.proposition_id == first.key
    assert third.key in supervisor.merge_propositions_view.actionable_keys()


def test_clicking_nonactionable_proposition_cancels_active_workspace(
    gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    first, _, third = supervisor.merge_propositions.catalog.propositions
    supervisor._activate_merge_proposition(supervisor.merge_propositions_view, first.key)
    supervisor.reject_merge_proposition()
    assert supervisor.selection.state.is_merge_mode

    view = supervisor.merge_propositions_view
    view._on_row_clicked(view._proxy_index_for_id(view._id_by_key[first.key]))

    assert not supervisor.selection.state.is_merge_mode
    assert view.current_key == first.key
    assert (
        supervisor.merge_propositions.catalog.status_for(first.key) is PropositionStatus.REJECTED
    )
    assert third.key in view.actionable_keys()


def test_merge_proposition_accept_returns_to_cluster_then_navigates_explicitly(
    monkeypatch, gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    first, overlap, next_proposition = supervisor.merge_propositions.catalog.propositions
    assignments_before = supervisor.clustering.spike_clusters.copy()
    supervisor._review_merge_proposition(supervisor.merge_propositions_view, first.key)
    workspace = supervisor.selection.snapshot()
    merge_view = supervisor.merge_view
    merge_dock = merge_view.dock
    supervisor.cluster_view.sort_by('id', 'asc')
    supervisor.cluster_view.filter('id >= 0')
    scrolled = []
    monkeypatch.setattr(
        supervisor.cluster_view,
        'scroll_to',
        lambda cluster_id, center: scrolled.append((cluster_id, center)),
    )

    merged_id = supervisor.merge().added[0]
    supervisor.block()

    assert (
        supervisor.merge_propositions.catalog.status_for(first.key) is PropositionStatus.ACCEPTED
    )
    assert supervisor.merge_propositions.catalog.status_for(overlap.key) is PropositionStatus.STALE
    assert supervisor.merge_propositions.catalog.reviews[first.key].applied_unit_ids == (30, 20)
    assert not supervisor.selection.state.is_merge_mode
    assert supervisor.selected_clusters == [merged_id]
    assert supervisor.selected_merge == []
    assert supervisor.merge_view is merge_view
    assert supervisor.merge_view.dock is merge_dock
    assert merge_dock.isHidden()
    assert supervisor.merge_propositions_view.current_key == first.key
    assert supervisor.cluster_view.filter_edit.text() == ''
    assert supervisor.cluster_view._current_sort == ('id', 'asc')
    assert scrolled == [(merged_id, True)]
    assert supervisor.actions.get('undo').isEnabled()
    assert supervisor.merge_propositions_view.select_key(overlap.key)
    assert not supervisor.merge_propositions_view.can_trigger('review')
    assignments_after = supervisor.clustering.spike_clusters.copy()

    supervisor.undo()
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, assignments_before)
    assert supervisor.selection.state == workspace
    assert supervisor.merge_propositions.catalog.status_for(first.key) is PropositionStatus.PENDING
    assert (
        supervisor.merge_propositions.catalog.status_for(overlap.key) is PropositionStatus.PENDING
    )
    assert supervisor.cluster_view.filter_edit.text() == 'id >= 0'

    supervisor.redo()
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, assignments_after)
    assert (
        supervisor.merge_propositions.catalog.status_for(first.key) is PropositionStatus.ACCEPTED
    )
    assert supervisor.merge_propositions.catalog.status_for(overlap.key) is PropositionStatus.STALE
    assert not supervisor.selection.state.is_merge_mode
    assert supervisor.selected_clusters == [merged_id]
    assert supervisor.selected_merge == []
    assert supervisor.merge_view.dock.isHidden()
    assert supervisor.cluster_view.filter_edit.text() == ''
    assert scrolled[-1] == (merged_id, True)

    supervisor.next_merge_proposition()

    assert supervisor.selection.state.merge.proposition_id == next_proposition.key
    assert supervisor.selected_merge == [11, 1]
    assert not supervisor.merge_view.dock.isHidden()


def test_final_merge_proposition_returns_to_quality_assignment(
    gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    supervisor = _proposition_supervisor(
        gui,
        cluster_ids,
        cluster_groups,
        cluster_labels,
        similarity,
        tempdir,
        merges=[{'unit_ids': [30, 20]}],
    )
    proposition = supervisor.merge_propositions.catalog.propositions[0]
    supervisor._review_merge_proposition(supervisor.merge_propositions_view, proposition.key)

    merged_id = supervisor.merge().added[0]
    supervisor.block()
    supervisor.move('good', 'all')

    assert not supervisor.selection.state.is_merge_mode
    assert supervisor.selected == [merged_id]
    assert supervisor.cluster_meta.get('group', merged_id) == 'good'
    assert supervisor.merge_propositions_view.current_key == proposition.key

    supervisor.next_merge_proposition()

    assert not supervisor.selection.state.is_merge_mode
    assert supervisor.selected == [merged_id]
    assert supervisor.merge_view.dock.isHidden()


def test_modified_proposition_accepts_numpy_candidate_and_restores_similarity_on_undo(
    gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    proposition = supervisor.merge_propositions.catalog.propositions[0]
    supervisor._review_merge_proposition(supervisor.merge_propositions_view, proposition.key)
    candidate = np.int64(supervisor.similarity_view.get_ids()[0])
    supervisor.add_to_merge((candidate,))
    workspace = supervisor.selection.snapshot()

    supervisor.merge()
    supervisor.block()
    review = supervisor.merge_propositions.catalog.reviews[proposition.key]
    assert all(type(cluster_id) is int for cluster_id in review.applied_unit_ids)

    supervisor.undo()
    supervisor.block()

    assert supervisor.selection.state == workspace
    assert set(supervisor.selected_merge).isdisjoint(supervisor.similarity_view.get_ids())
    supervisor.next()
    supervisor.block()
    assert set(supervisor.selected_merge).isdisjoint(supervisor.selected_similar)


def test_proposition_acceptance_preflight_failure_is_atomic(
    monkeypatch, gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    proposition = supervisor.merge_propositions.catalog.propositions[0]
    supervisor._review_merge_proposition(supervisor.merge_propositions_view, proposition.key)
    assignments = supervisor.clustering.spike_clusters.copy()
    workspace = supervisor.selection.snapshot()
    history_position = supervisor._global_history.current_position

    def fail(*args, **kwargs):
        raise RuntimeError('review preflight failed')

    monkeypatch.setattr(supervisor.merge_propositions, 'prepare_accept', fail)
    with raises(RuntimeError, match='review preflight failed'):
        supervisor.merge()

    ae(supervisor.clustering.spike_clusters, assignments)
    assert supervisor.selection.state is workspace
    assert supervisor._global_history.current_position == history_position
    assert (
        supervisor.merge_propositions.catalog.status_for(proposition.key)
        is PropositionStatus.PENDING
    )


def test_failed_proposition_merge_and_reject_history(
    monkeypatch, gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    key = supervisor.merge_propositions.catalog.propositions[0].key
    supervisor._review_merge_proposition(supervisor.merge_propositions_view, key)
    workspace = supervisor.selection.snapshot()
    merge_view = supervisor.merge_view

    def fail(*args, **kwargs):
        raise RuntimeError('merge failed')

    monkeypatch.setattr(supervisor.clustering, 'merge', fail)
    with raises(RuntimeError, match='merge failed'):
        supervisor.merge()
    assert supervisor.selection.state is workspace
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.PENDING

    supervisor.reject_merge_proposition()
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.REJECTED
    next_key = supervisor.merge_propositions.catalog.propositions[1].key
    assert supervisor.selection.state.merge.proposition_id == next_key
    assert supervisor.merge_view is merge_view
    assert not merge_view.dock.isHidden()
    assert supervisor.actions.get('undo').isEnabled()
    supervisor.undo()
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.PENDING
    assert supervisor.selection.state == workspace
    supervisor.redo()
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.REJECTED
    assert supervisor.selection.state.merge.proposition_id == next_key

    supervisor._activate_merge_proposition(supervisor.merge_propositions_view, key)
    assert not supervisor.selection.state.is_merge_mode
    supervisor.reset_merge_proposition()
    assert supervisor.merge_propositions.catalog.status_for(key) is PropositionStatus.PENDING
    assert supervisor.selection.state.merge.proposition_id == key


def test_supervisor_close_releases_owned_event_callbacks(
    gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
):
    supervisor = _proposition_supervisor(
        gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
    )
    key = supervisor.merge_propositions.catalog.propositions[0].key
    supervisor._review_merge_proposition(supervisor.merge_propositions_view, key)
    views = (
        supervisor.cluster_view,
        supervisor.similarity_view,
        supervisor.merge_view,
        supervisor.merge_propositions_view,
    )
    owned = {
        gui,
        supervisor,
        supervisor.action_creator,
        supervisor.clustering,
        supervisor.cluster_meta,
        supervisor.merge_propositions,
        *views,
        *(view.dock for view in views),
    }

    gui.close()

    assert supervisor._merge_close_callback is None
    assert not any(
        sender in owned or getattr(callback, '__self__', None) in owned
        for _, sender, callback, _ in _EVENT._callbacks
    )


def test_saving_gui_state_cancels_transient_merge_selection(supervisor):
    _select(supervisor, [30], [20])
    entry = supervisor.selection.snapshot()
    supervisor.toggle_merge_mode()

    supervisor._save_gui_state(supervisor.gui)

    assert supervisor.selection.state == entry
    assert supervisor.merge_view is None


def test_stale_table_selection_revision_is_ignored(supervisor):
    _select(supervisor, [10], [20])
    state = supervisor.selection.state

    supervisor._clusters_selected(
        supervisor.cluster_view,
        {
            'selected': [30],
            'next': None,
            'kwargs': {},
            'revision': supervisor.cluster_view._selection_revision - 1,
        },
    )

    assert supervisor.selection.state is state


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


def test_supervisor_multi_cluster_reference_is_explicit_and_blue(supervisor):
    requested = []

    @connect(sender=supervisor.similarity_view)
    def on_request_similar_clusters(sender, cluster_id):
        requested.append(cluster_id)

    _select(supervisor, [10, 30], [20])

    # The first Cluster View row is the explicit Similarity reference and owns
    # the blue positional color slot.
    assert requested == [10]
    assert supervisor.selected == [10, 30, 20]

    def rgb(color):
        return tuple(channel / 255 for channel in color.getRgb()[:3])

    def expected_rgb(index):
        return tuple(int(channel * 255) / 255 for channel in selected_cluster_color(index)[:3])

    assert rgb(supervisor.cluster_view._selection_background(10)) == expected_rgb(0)
    assert rgb(supervisor.cluster_view._selection_background(30)) == expected_rgb(1)
    assert rgb(supervisor.similarity_view._selection_background(20)) == expected_rgb(2)

    unconnect(on_request_similar_clusters)


def test_normal_presentation_follows_table_order_and_resorting(supervisor):
    _select(supervisor, [30])
    similarity_view = supervisor.similarity_view
    similarity_view.sort_by('id', 'asc')

    # Select out of row order: presentation and positional colors still follow the table.
    similarity_view.select([20, 1, 11])
    supervisor.block()
    assert supervisor.selected_similar == [20, 1, 11]
    assert supervisor.selected == [30, 1, 11, 20]
    assert similarity_view._selected_color_index(1) == 1
    assert similarity_view._selected_color_index(11) == 2
    assert similarity_view._selected_color_index(20) == 3

    events = []

    @connect(sender=supervisor)
    def on_select(sender, cluster_ids):
        events.append(cluster_ids)

    similarity_view.sort_by('id', 'desc')

    assert supervisor.selected == [30, 20, 11, 1]
    assert events == [[30, 20, 11, 1]]
    assert similarity_view._selected_color_index(20) == 3
    assert similarity_view._selected_color_index(11) == 2
    assert similarity_view._selected_color_index(1) == 1
    unconnect(on_select)


def test_normal_similarity_insertion_does_not_recolor_existing_rows(supervisor):
    _select(supervisor, [30])
    similarity_view = supervisor.similarity_view
    similarity_view.sort_by('id', 'asc')

    similarity_view.select([1])
    supervisor.block()
    similarity_view.select_toggle(20)
    supervisor.block()
    color_before = similarity_view._selected_color_index(20)

    # Insert 11 before 20 in visible row order without changing 20's color slot.
    similarity_view.select_toggle(11)
    supervisor.block()

    assert supervisor.selected == [30, 1, 11, 20]
    assert similarity_view._selected_color_index(20) == color_before
    assert similarity_view._selected_color_index(11) > color_before


def test_direct_similarity_replacements_reuse_first_candidate_color(supervisor):
    _select(supervisor, [30])
    similarity_view = supervisor.similarity_view

    for candidate in (20, 1, 11):
        similarity_view.select([candidate])
        supervisor.block()

        assert supervisor.selected_similar == [candidate]
        assert similarity_view._selected_color_index(candidate) == 1


def test_normal_similarity_deselection_and_reselection_preserve_color_slots(supervisor):
    _select(supervisor, [30])
    similarity_view = supervisor.similarity_view
    similarity_view.sort_by('id', 'asc')
    similarity_view.select([1, 11, 20])
    supervisor.block()
    colors_before = {
        cluster_id: similarity_view._selected_color_index(cluster_id) for cluster_id in (1, 11, 20)
    }

    similarity_view.select_toggle(11)
    supervisor.block()
    assert {
        cluster_id: similarity_view._selected_color_index(cluster_id) for cluster_id in (1, 20)
    } == {cluster_id: colors_before[cluster_id] for cluster_id in (1, 20)}

    similarity_view.select_toggle(11)
    supervisor.block()
    assert {
        cluster_id: similarity_view._selected_color_index(cluster_id) for cluster_id in (1, 11, 20)
    } == colors_before


def test_merge_presentation_keeps_merge_order_before_similarity_table_order(supervisor):
    _select(supervisor, [30])
    supervisor.toggle_merge_mode()
    similarity_view = supervisor.similarity_view
    similarity_view.sort_by('id', 'asc')

    similarity_view.select([20, 1, 11])
    supervisor.block()
    assert supervisor.selected == [30, 1, 11, 20]

    supervisor.add_to_merge((11,), insertion=1)
    assert supervisor.selected_merge == [30, 11]
    assert supervisor.selected == [30, 11, 1, 20]

    similarity_view.sort_by('id', 'desc')
    assert supervisor.selected_merge == [30, 11]
    assert supervisor.selected == [30, 11, 20, 1]


def test_table_filter_reorders_normal_presentation_without_recoloring(supervisor):
    _select(supervisor, [30])
    similarity_view = supervisor.similarity_view
    similarity_view.sort_by('id', 'asc')
    # Click A, C, B, while table order establishes A, B, C presentation.
    similarity_view.select([1, 20, 11])
    supervisor.block()
    assert supervisor.selected == [30, 1, 11, 20]
    colors = dict(supervisor.selection_color_indices)
    roles = (supervisor.selected_clusters, supervisor.selected_similar)
    events = []

    @connect(sender=supervisor)
    def on_select(sender, cluster_ids):
        events.append(cluster_ids)

    # Retain only A. Hidden B/C must use the prior presentation order, not
    # the Similarity role's click order (A, C, B).
    similarity_view.filter('id < 2')

    assert supervisor.selected == [30, 1, 11, 20]
    assert dict(supervisor.selection_color_indices) == colors
    assert (supervisor.selected_clusters, supervisor.selected_similar) == roles
    assert events == []
    unconnect(on_select)


def test_filtered_ctrl_toggle_preserves_hidden_selection_and_colors(supervisor):
    _select(supervisor, [30])
    similarity_view = supervisor.similarity_view
    similarity_view.sort_by('id', 'asc')
    similarity_view.select([1, 11, 20])
    supervisor.block()
    colors = dict(supervisor.selection_color_indices)
    similarity_view.filter('id < 2')

    similarity_view.select_toggle(1)
    supervisor.block()

    assert supervisor.selected_similar == [11, 20]
    assert {
        cluster_id: supervisor.selection_color_indices[cluster_id] for cluster_id in colors
    } == colors

    similarity_view.select_toggle(1)
    supervisor.block()

    assert set(supervisor.selected_similar) == {1, 11, 20}
    assert dict(supervisor.selection_color_indices) == colors


def test_table_filter_reorders_merge_similarity_tail_without_recoloring(supervisor):
    _select(supervisor, [30])
    supervisor.toggle_merge_mode()
    similarity_view = supervisor.similarity_view
    similarity_view.sort_by('id', 'asc')
    similarity_view.select([1, 20, 11])
    supervisor.block()
    assert supervisor.selected == [30, 1, 11, 20]
    colors = dict(supervisor.selection_color_indices)
    roles = (supervisor.selected_merge, supervisor.selected_similar)
    events = []

    @connect(sender=supervisor)
    def on_select(sender, cluster_ids):
        events.append(cluster_ids)

    similarity_view.filter('id < 2')

    assert supervisor.selected == [30, 1, 11, 20]
    assert dict(supervisor.selection_color_indices) == colors
    assert (supervisor.selected_merge, supervisor.selected_similar) == roles
    assert events == []
    unconnect(on_select)


def test_supervisor_select_event_has_legacy_payload_and_suppression(supervisor):
    events = []

    @connect(sender=supervisor)
    def on_select(sender, cluster_ids, **kwargs):
        events.append((sender, cluster_ids, kwargs))

    supervisor.cluster_view.select([10, 30], marker='legacy')
    supervisor.block()

    assert events == [(supervisor, [10, 30], {'marker': 'legacy'})]

    # ``update_views`` is an internal suppression flag: it neither reaches
    # public listeners nor changes the selected rows.
    supervisor.cluster_view.select([20], update_views=False)
    supervisor.block()
    assert events == [(supervisor, [10, 30], {'marker': 'legacy'})]
    assert supervisor.selected_clusters == [20]

    unconnect(on_select)


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

    # The shortcut variant uses the saved preference and advances past the current selection.
    similarity_view.sort_by('id', 'desc')
    navigable_ids = similarity_view.get_navigable_ids()
    previous_last = navigable_ids.index(similarity_view.get_selected_ids()[-1])
    control_modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier
    qtbot.keyClick(gui, Qt.Key_Space, control_modifier)
    supervisor.block()
    assert supervisor.selected_clusters == [30]
    assert supervisor.selected_similar == navigable_ids[previous_last + 1 : previous_last + 3]

    # Selecting more rows than are available is safe.
    supervisor.select_actions.select_n_similar(100)
    supervisor.block()
    assert supervisor.selected_similar == navigable_ids

    # The preference is stored in global GUI state.
    supervisor._save_gui_state(gui)
    assert gui.state['n_similar_clusters_to_select'] == 100


def test_select_first_similar_settles_pending_mutations_in_merge_mode(qtbot, supervisor):
    _select(supervisor, [30])
    supervisor.toggle_merge_mode()
    view = supervisor.similarity_view
    candidate = view.get_navigable_ids()[0]

    view.debouncer._last_submission_time = default_timer()
    view.select([candidate])
    assert view.debouncer.has_pending
    supervisor.select_first_similar()
    supervisor.select_first_similar()
    qtbot.wait(1)
    supervisor.block()

    assert tuple(view.get_selected_ids()) == supervisor.selection.state.similar_ids
    assert set(supervisor.selected_merge).isdisjoint(supervisor.selected_similar)


@mark.parametrize('filtered', [False, True])
@mark.parametrize('which', ['best', 'all'])
def test_batch_selection_after_moving_last_primary_is_empty(supervisor, filtered, which):
    supervisor.cluster_view.sort_by('id', 'asc')
    last = supervisor.cluster_view.get_navigable_ids()[-1]
    _select(supervisor, [last])
    if filtered:
        supervisor.cluster_view.filter(f'id == {last} and group != "good"')
    assert supervisor.similarity_view.get_navigable_ids()
    if which == 'all':
        supervisor.similarity_view.select(supervisor.similarity_view.get_navigable_ids()[:1])
        supervisor.block()

    getattr(supervisor.actions, f'move_{which}_to_good')()
    supervisor.block()

    assert supervisor.selection.state.reference_id is None
    assert supervisor.similarity_view.get_ids() == []
    supervisor.select_actions.select_first_similar()
    supervisor.block()
    assert supervisor.selected == []
    assert supervisor.similarity_view.get_selected_ids() == []
    assert not supervisor.task_logger._processing


@mark.parametrize('context', ['reference', 'clear', 'filter', 'model', 'merge_mode', 'close'])
def test_batch_selection_has_no_deferred_context_mutation(supervisor, monkeypatch, context):
    _select(supervisor, [30])
    view = supervisor.similarity_view
    expected = view.get_navigable_ids()[: supervisor.n_similar_clusters_to_select]
    scheduled, completed = [], []

    def schedule(callback, *args):
        if callback is not None:
            scheduled.append((callback, args))

    monkeypatch.setattr(view, '_schedule_callback', schedule)
    supervisor.select_first_similar(callback=completed.append)
    view.debouncer.flush()
    # Only completion is deferred. No callback may create a selection from old rows.
    assert supervisor.selected_similar == expected
    assert not completed
    assert len(scheduled) == 1
    if context == 'reference':
        supervisor.select([20])
        supervisor.cluster_view.debouncer.flush()
    elif context == 'clear':
        supervisor.select([])
        supervisor.cluster_view.debouncer.flush()
    elif context == 'filter':
        view.filter('id < 2')
    elif context == 'model':
        supervisor.actions.merge()
    elif context == 'merge_mode':
        supervisor.toggle_merge_mode()
    else:
        supervisor.gui.close()
    settled = supervisor.selection.snapshot()
    for callback, args in scheduled:
        callback(*args)
    assert supervisor.selection.state == settled
    assert len(completed) == 1


def test_similarity_reset_without_primary_clears_candidates(supervisor):
    _select(supervisor, [30], [20])
    view = supervisor.similarity_view
    view.reset(())
    assert view.get_ids() == []
    assert view.get_selected_ids() == []


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
    assert (
        _get_shortcut_string(supervisor.select_actions.get('toggle_merge_mode').shortcut()) == 'v'
    )
    toolbar_actions = gui._toolbar.actions()
    assert supervisor.action_creator.edit_actions.get('merge') in toolbar_actions
    assert supervisor.action_creator.select_actions.get('toggle_merge_mode') in toolbar_actions
    assert gui.help_actions.get('show_all_shortcuts') in toolbar_actions
    save_index = toolbar_actions.index(gui.file_actions.get('save'))
    assert toolbar_actions[save_index - 6 : save_index] == [
        supervisor.action_creator.select_actions.get('toggle_merge_mode'),
        supervisor.action_creator.edit_actions.get('merge'),
        toolbar_actions[save_index - 4],
        supervisor.action_creator.edit_actions.get('undo'),
        supervisor.action_creator.edit_actions.get('redo'),
        toolbar_actions[save_index - 1],
    ]
    assert toolbar_actions[save_index - 4].isSeparator()
    assert toolbar_actions[save_index - 1].isSeparator()
    assert toolbar_actions[save_index + 1].isSeparator()
    assert not supervisor.action_creator.select_actions.get('toggle_merge_mode').icon().isNull()
    assert not supervisor.action_creator.edit_actions.get('merge').icon().isNull()
    assert not gui.help_actions.get('show_all_shortcuts').icon().isNull()

    select_menu = gui.get_menu('Sele&ct')
    navigation_menu = next(
        action.menu() for action in select_menu.actions() if action.text() == 'Navigation'
    )
    navigation_actions = [
        action for action in navigation_menu.actions() if not action.isSeparator()
    ]
    assert navigation_actions == [
        supervisor.select_actions.get(name)
        for name in (
            'first',
            'last',
            'reset_wizard',
            'next',
            'previous',
            'next_best',
            'previous_best',
        )
    ]

    with raises(ValueError, match='positive integer'):
        supervisor.select_first_similar(0)
    with raises(ValueError, match='positive integer'):
        supervisor.select_first_similar(1.5)


def test_supervisor_restores_independent_table_column_orders(gui, cluster_ids, similarity):
    gui.state['table_column_orders'] = {
        'cluster': ['n_spikes', 'id'],
        'similarity': ['similarity', 'id', 'n_spikes'],
        'merge': ['n_spikes', 'id'],
    }
    supervisor = Supervisor(np.repeat(cluster_ids, 2), similarity=similarity)
    supervisor.attach(gui)

    assert supervisor.cluster_view.column_order()[:2] == ['n_spikes', 'id']
    assert supervisor.similarity_view.column_order()[:3] == ['similarity', 'id', 'n_spikes']
    supervisor.select([10, 20])
    supervisor.block()
    supervisor.toggle_merge_mode()
    assert supervisor.merge_view.column_order()[:2] == ['n_spikes', 'id']


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
    assert not gui.windowTitle().startswith('* ')
    assert not gui.file_actions.get('save').isEnabled()
    supervisor.label('group', 'noise', [30])
    supervisor.block()
    assert gui.windowTitle().startswith('* ')
    assert gui.file_actions.get('save').isEnabled()

    emit('request_save', gui)
    assert gui.status_message == 'Curation changes saved.'
    assert not gui.windowTitle().startswith('* ')
    assert not gui.file_actions.get('save').isEnabled()


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
    selection_before = supervisor.selection.snapshot()

    supervisor.actions.merge()
    supervisor.block()

    _assert_selected(supervisor, [31])
    selection_after = supervisor.selection.snapshot()

    supervisor.actions.undo()
    supervisor.block()
    _assert_selected(supervisor, [30, 20])
    assert supervisor.selection.state == selection_before

    supervisor.actions.redo()
    supervisor.block()
    supervisor.task_logger.show_history()
    _assert_selected(supervisor, [31])
    assert supervisor.selection.state == selection_after

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


def test_supervisor_redo_preserves_selection_exploration_after_action(supervisor):
    _select(supervisor, [30], [20])
    selection_before = supervisor.selection.snapshot()
    supervisor.actions.merge()
    supervisor.block()

    next_similar = supervisor.similarity_view.get_ids()[0]
    supervisor.similarity_view.select([next_similar])
    supervisor.block()
    selection_at_undo = supervisor.selection.snapshot()

    supervisor.actions.undo()
    supervisor.block()
    assert supervisor.selection.state == selection_before

    supervisor.actions.redo()
    supervisor.block()
    assert supervisor.selection.state == selection_at_undo


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
    selection_before = supervisor.selection.snapshot()

    supervisor.actions.split([1, 2])
    supervisor.block()

    _assert_selected(supervisor, [31, 33, 32])
    selection_after = supervisor.selection.snapshot()

    supervisor.actions.undo()
    supervisor.block()
    _assert_selected(supervisor, [1, 2])
    assert supervisor.selection.state == selection_before

    supervisor.actions.redo()
    supervisor.block()
    _assert_selected(supervisor, [31, 33, 32])
    assert supervisor.selection.state == selection_after


def test_supervisor_split_1(supervisor):
    supervisor.select_actions.select([1, 2])
    supervisor.block()

    @connect(sender=supervisor)
    def on_request_split(sender):
        return [1, 2]

    supervisor.actions.split()
    supervisor.block()
    _assert_selected(supervisor, [31, 33, 32])


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
    selection_before = supervisor.selection.snapshot()

    assert not supervisor.move('', '')

    supervisor.actions.move('noise', 'all')
    supervisor.block()
    _assert_selected(supervisor, [11])
    selection_after = supervisor.selection.snapshot()

    supervisor.actions.undo()
    supervisor.block()
    _assert_selected(supervisor, [20])
    assert supervisor.selection.state == selection_before

    supervisor.actions.redo()
    supervisor.block()
    _assert_selected(supervisor, [11])
    assert supervisor.selection.state == selection_after


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
    assert supervisor.similarity_view._selected_color_index(20) == 1

    supervisor.select_actions.next()
    supervisor.block()
    _assert_selected(supervisor, [30, 11])
    assert supervisor.similarity_view._selected_color_index(11) == 1

    supervisor.select_actions.previous()
    supervisor.block()
    _assert_selected(supervisor, [30, 20])
    assert supervisor.similarity_view._selected_color_index(20) == 1

    supervisor.select_actions.unselect_similar()
    supervisor.block()
    _assert_selected(supervisor, [30])

    supervisor.select_actions.next()
    supervisor.block()
    assert supervisor.similarity_view._selected_color_index(supervisor.selected_similar[0]) == 1


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


@mark.parametrize('merge_mode', [False, True])
def test_debounced_similarity_batch_matches_sequential_reduction(supervisor, merge_mode):
    from phy.cluster._selection import CurationSelectionController

    _select(supervisor, [30])
    if merge_mode:
        supervisor.toggle_merge_mode()
    view = supervisor.similarity_view
    view.select([1, 11, 20])
    supervisor.block()
    view.sort_by('id', 'asc')
    initial = supervisor.selection.state
    events = []

    @connect(event='select', sender=supervisor)
    def selected(sender, cluster_ids, **kwargs):
        events.append((cluster_ids, kwargs))

    try:
        view.debouncer.delay = 60_000
        view.debouncer._last_submission_time = default_timer()
        view.select_toggle(11)
        view.next()
        view.select_until(20)
        view.select_toggle(1)
        payload = next(iter(view.debouncer.pending_functions.values()))[1][0]
        operations = payload['kwargs']['_selection_mutations']
        expected = CurationSelectionController(initial)
        expected.apply_similarity_mutations(operations, supervisor._presentation_order_from_tables)
        view.debouncer.flush()
        assert supervisor.selection.state == expected.state
        assert len(events) == 1
        assert events[0][1] == {}
        assert tuple(view._selected_ids) == supervisor.selection.state.similar_ids
    finally:
        unconnect(selected)


def test_pending_similarity_batch_retains_hidden_ids_and_reference_reset(supervisor):
    _select(supervisor, [30], [1, 11, 20])
    view = supervisor.similarity_view
    view.filter('id != 11')
    view.debouncer.delay = 60_000
    view.debouncer._last_submission_time = default_timer()
    view.select_toggle(1)
    view.select_toggle(20)
    view.debouncer.flush()
    assert supervisor.selection.state.similar_ids == (11,)
    assert view.get_selected_ids() == []

    view.select_toggle(1)
    view.select_toggle(20)
    supervisor.cluster_view.select([10])
    supervisor.cluster_view.debouncer.flush()
    view.debouncer.flush()
    assert supervisor.selection.state.cluster_ids == (10,)
    assert supervisor.selection.state.similar_ids == ()
    assert view._pending_selection_mutations == []


def test_rapid_next_actions_keep_ordered_selection_chain(qtbot, supervisor):
    _select(supervisor, [30], [1])
    view = supervisor.similarity_view
    view.sort_by('id', 'asc')
    view.debouncer.delay = 60_000
    view.debouncer._last_submission_time = default_timer()
    for _ in range(3):
        supervisor.select_actions.next()
        qtbot.wait(1)
    assert view.debouncer.has_pending
    assert supervisor.selected_similar == [1]
    view.debouncer.flush()
    assert supervisor.selected_similar == view.get_selected_ids()
    assert not supervisor.task_logger._processing


def test_settle_selection_drains_observer_generated_operations(supervisor):
    _select(supervisor, [30], [1])
    view = supervisor.similarity_view
    view.debouncer.delay = 60_000
    view.debouncer._last_submission_time = default_timer()
    view.select([11])
    calls = []

    @connect(event='select', sender=supervisor)
    def selected(sender, cluster_ids):
        calls.append(cluster_ids)
        if len(calls) == 1:
            view.select([20])

    try:
        supervisor._settle_selection()
        assert supervisor.selected_similar == [20]
        assert not view.debouncer.has_pending
        assert calls == [[30, 11], [30, 20]]
    finally:
        unconnect(selected)


def test_settle_selection_rejects_reentrant_curation(supervisor):
    _select(supervisor, [30], [1])
    view = supervisor.similarity_view
    view.debouncer.delay = 60_000
    view.debouncer._last_submission_time = default_timer()
    view.select([11])
    assignments = supervisor.clustering.spike_clusters.copy()

    @connect(event='select', sender=supervisor)
    def selected(sender, cluster_ids):
        supervisor.merge()

    try:
        with raises(RuntimeError, match='selection observers'):
            supervisor._settle_selection()
        ae(supervisor.clustering.spike_clusters, assignments)
        assert not supervisor._settling_selection
    finally:
        unconnect(selected)


def test_role_transfer_settles_other_pending_similarity_members(supervisor):
    _select(supervisor, [30], [1])
    view = supervisor.similarity_view
    view.debouncer.delay = 60_000
    view.debouncer._last_submission_time = default_timer()
    view.select_toggle(11)
    supervisor.transfer_cluster(1)
    assert supervisor.selection.state.cluster_ids == (30, 1)
    assert supervisor.selection.state.similar_ids == (11,)
    view.debouncer.flush()
    assert supervisor.selection.state.similar_ids == (11,)


def test_never_settling_selection_observer_blocks_curation(supervisor):
    _select(supervisor, [30], [1])
    view = supervisor.similarity_view
    view.debouncer.delay = 60_000
    view.debouncer._last_submission_time = default_timer()
    view.select([11])
    assignments = supervisor.clustering.spike_clusters.copy()

    @connect(event='select', sender=supervisor)
    def selected(sender, cluster_ids):
        view.select([1] if supervisor.selected_similar == [11] else [11])

    try:
        with raises(RuntimeError, match='did not settle'):
            supervisor.merge()
        ae(supervisor.clustering.spike_clusters, assignments)
        assert not supervisor._settling_selection
    finally:
        unconnect(selected)
        view.debouncer.flush()


def test_multi_suggestions_compare_union_then_accept_atomically(
    gui,
    cluster_ids,
    cluster_groups,
    cluster_labels,
    similarity,
    tempdir,
):
    supervisor = _proposition_supervisor(
        gui,
        cluster_ids,
        cluster_groups,
        cluster_labels,
        similarity,
        tempdir,
    )
    first, overlap, _ = supervisor.merge_propositions.catalog.propositions
    keys = (first.key, overlap.key)
    _select(supervisor, [11], [10])
    original = supervisor.clustering.spike_clusters.copy()
    view = supervisor.merge_propositions_view
    supervisor._activate_merge_propositions(view, keys)
    assert supervisor.selected == [30, 20, 10]
    assert supervisor.selection.state.merge.proposition_ids == keys
    assert view.selected_keys == keys
    assert not supervisor.merge_propositions.catalog.reviews
    np.testing.assert_array_equal(supervisor.clustering.spike_clusters, original)
    workspace = supervisor.selection.snapshot()
    supervisor._refresh_propositions()
    assert view.selected_keys == keys
    supervisor.reject_merge_proposition()
    assert supervisor.selection.snapshot() == workspace
    assert not supervisor.merge_propositions.catalog.reviews
    merged = supervisor.merge().added[0]
    for key in keys:
        review = supervisor.merge_propositions.catalog.reviews[key]
        assert review.result_unit_id == merged
        assert review.applied_unit_ids == (30, 20, 10)
    supervisor.undo()
    supervisor.block()
    np.testing.assert_array_equal(supervisor.clustering.spike_clusters, original)
    assert not supervisor.merge_propositions.catalog.reviews
    assert supervisor.selection.snapshot() == workspace
    assert view.selected_keys == keys
    supervisor.redo()
    supervisor.block()
    assert len(supervisor.merge_propositions.catalog.reviews) == 2
    supervisor._activate_merge_propositions(view, ())
    assert not supervisor.selection.state.is_merge_mode


def test_multi_suggestion_cancel_restores_entry_and_rejects_stale(
    gui,
    cluster_ids,
    cluster_groups,
    cluster_labels,
    similarity,
    tempdir,
):
    supervisor = _proposition_supervisor(
        gui,
        cluster_ids,
        cluster_groups,
        cluster_labels,
        similarity,
        tempdir,
    )
    _select(supervisor, [11], [10])
    entry = supervisor.selection.snapshot()
    first, overlap, _ = supervisor.merge_propositions.catalog.propositions
    view = supervisor.merge_propositions_view
    supervisor._activate_merge_propositions(view, (first.key, overlap.key))
    supervisor._activate_merge_propositions(view, ())
    assert supervisor.selection.snapshot() == entry
    assert not supervisor.merge_propositions.catalog.reviews
    supervisor.merge_propositions.reject(overlap.key)
    supervisor._activate_merge_propositions(view, (first.key, overlap.key))
    assert supervisor.selection.snapshot() == entry
    assert supervisor.merge_propositions.catalog.status_for(first.key) is PropositionStatus.PENDING


def test_multi_suggestion_preflight_failure_does_not_change_assignments(
    monkeypatch,
    gui,
    cluster_ids,
    cluster_groups,
    cluster_labels,
    similarity,
    tempdir,
):
    supervisor = _proposition_supervisor(
        gui,
        cluster_ids,
        cluster_groups,
        cluster_labels,
        similarity,
        tempdir,
    )
    keys = tuple(p.key for p in supervisor.merge_propositions.catalog.propositions[:2])
    supervisor._activate_merge_propositions(supervisor.merge_propositions_view, keys)
    original = supervisor.clustering.spike_clusters.copy()
    snapshot = supervisor.selection.snapshot()

    def fail(*args):
        raise ValueError('review preflight failed')

    monkeypatch.setattr(supervisor.merge_propositions, 'prepare_accept_many', fail)
    with raises(ValueError, match='review preflight failed'):
        supervisor.merge()
    np.testing.assert_array_equal(supervisor.clustering.spike_clusters, original)
    assert supervisor.selection.snapshot() == snapshot
    assert not supervisor.merge_propositions.catalog.reviews
