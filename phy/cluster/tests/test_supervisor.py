"""Test GUI component."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

# from contextlib import contextmanager

import sys

import numpy as np
from numpy.testing import assert_array_equal as ae
from phylib.utils import Bunch, connect, emit, unconnect
from phylib.utils.event import _EVENT
from pytest import fixture, raises

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


def test_cluster_view_control_right_click_reports_unselected_row_without_selecting_it(
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
    control_modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier
    qtbot.mouseClick(cv.table_view.viewport(), Qt.RightButton, control_modifier, pos=pos)

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
    assert supervisor.merge_view._drag_ids_for_index(reference) == ()

    supervisor.merge_view.emit_cluster_drop(supervisor.similarity_view, (candidate,), 1)
    assert supervisor.selected_merge == [10, candidate, 30, 20]

    supervisor.merge_view.emit_cluster_drop(supervisor.merge_view, (20,), 1)
    assert supervisor.selected_merge == [10, 20, candidate, 30]

    supervisor.similarity_view.emit_cluster_drop(supervisor.merge_view, (candidate,), 0)
    assert supervisor.selected_merge == [10, 20, 30]
    assert supervisor.selected_similar == [candidate]

    supervisor.toggle_merge_mode()
    assert not supervisor.similarity_view.table_view.dragEnabled()


def test_supervisor_control_right_click_transfers_only_in_merge_mode(qtbot, supervisor):
    _select(supervisor, [10, 30], [20])
    candidate = next(
        cluster_id
        for cluster_id in supervisor.similarity_view.get_ids()
        if cluster_id not in supervisor.selected_similar
    )
    index = supervisor.similarity_view._proxy_index_for_id(candidate)
    pos = supervisor.similarity_view.table_view.visualRect(index).center()
    control_modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier

    qtbot.mouseClick(
        supervisor.similarity_view.table_view.viewport(),
        Qt.RightButton,
        control_modifier,
        pos=pos,
    )
    supervisor.block()
    assert candidate not in supervisor.selected_clusters

    supervisor.toggle_merge_mode()
    index = supervisor.similarity_view._proxy_index_for_id(candidate)
    pos = supervisor.similarity_view.table_view.visualRect(index).center()

    qtbot.mouseClick(
        supervisor.similarity_view.table_view.viewport(),
        Qt.RightButton,
        control_modifier,
        pos=pos,
    )
    supervisor.block()
    assert candidate in supervisor.selected_merge

    supervisor._remove_merge_candidate_on_right_click(supervisor.merge_view, candidate)
    supervisor.block()
    assert candidate not in supervisor.selected_merge
    assert candidate in supervisor.selected_similar


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
    assert supervisor.selected == [merged_id]
    merge_view = supervisor.merge_view
    assert merge_view is not None
    assert merge_view.dock.isHidden()
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
    assert supervisor.selected == [merged_id]
    assert supervisor.merge_view is merge_view
    assert merge_view.dock.isHidden()
    assert events[-1] == [merged_id]
    unconnect(on_select)


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


def _proposition_supervisor(gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir):
    catalog = decode_curation_mapping(
        {
            'format_version': '2',
            'unit_ids': cluster_ids,
            'merges': [
                {'unit_ids': [30, 20]},
                {'unit_ids': [20, 10]},
                {'unit_ids': [11, 1]},
            ],
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


def test_merge_proposition_accept_overlap_and_coupled_undo_redo(
    gui, cluster_ids, cluster_groups, cluster_labels, similarity, tempdir
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

    supervisor.merge()
    supervisor.block()

    assert (
        supervisor.merge_propositions.catalog.status_for(first.key) is PropositionStatus.ACCEPTED
    )
    assert supervisor.merge_propositions.catalog.status_for(overlap.key) is PropositionStatus.STALE
    assert supervisor.merge_propositions.catalog.reviews[first.key].applied_unit_ids == (30, 20)
    assert supervisor.selection.state.merge.proposition_id == next_proposition.key
    assert supervisor.selected_merge == [11, 1]
    assert supervisor.merge_view is merge_view
    assert supervisor.merge_view.dock is merge_dock
    assert not merge_dock.isHidden()
    assert supervisor.merge_propositions_view.current_key == next_proposition.key
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

    supervisor.redo()
    supervisor.block()
    ae(supervisor.clustering.spike_clusters, assignments_after)
    assert (
        supervisor.merge_propositions.catalog.status_for(first.key) is PropositionStatus.ACCEPTED
    )
    assert supervisor.merge_propositions.catalog.status_for(overlap.key) is PropositionStatus.STALE
    assert supervisor.selection.state.merge.proposition_id == next_proposition.key
    assert supervisor.selected_merge == [11, 1]


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
