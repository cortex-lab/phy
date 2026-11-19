"""Metadata edits update table semantics without rebuilding scientific evidence."""

import numpy as np
import pytest
from phylib.utils import connect, emit, unconnect

from phy.apps.large_recording.tests.test_adapter import make_controller


def test_label_undo_redo_keep_scientific_artists_and_refresh_only_changed_rows(
    qtbot, tmp_path, monkeypatch
):
    from phy.apps.large_recording.adapter import BoundedTemplateController

    builtin_calls = []
    channel_label = BoundedTemplateController.get_best_channel_label

    def measured_channel_label(self, cluster):
        builtin_calls.append(cluster)
        return channel_label(self, cluster)

    monkeypatch.setattr(
        BoundedTemplateController, 'get_best_channel_label', measured_channel_label
    )
    _, controller = make_controller(tmp_path)
    supervisor = controller.supervisor
    metrics = []

    def quality_metric(cluster):
        metrics.append(cluster)
        return supervisor.cluster_meta.get('group', cluster)

    supervisor.cluster_metrics['quality_metric'] = quality_metric
    supervisor.columns.append('quality_metric')
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    events = []

    @connect(event='bounded_state_changed', sender=supervisor)
    def on_state(sender):
        events.append('bounded')

    @connect(event='cluster', sender=supervisor)
    def on_cluster(sender, up):
        events.append('cluster')

    try:
        supervisor.select([0])
        supervisor.block()
        view = gui.get_view('UnitStabilityView')
        view.on_select([0])
        data = view.get_data(0)
        artists = tuple(
            artist for ax in view.canvas.iter_ax() for artist in (*ax.lines, *ax.patches)
        )
        selected = supervisor.selection.snapshot()
        metrics.clear()
        events.clear()

        def forbid(*args, **kwargs):
            raise AssertionError('Unchanged scientific evidence was invalidated or redrawn.')

        monkeypatch.setattr(view, 'plot', forbid)
        monkeypatch.setattr(view, '_invalidate', forbid)
        patches = []
        change = supervisor.cluster_view.change

        def record(objects):
            patches.extend(objects)
            return change(objects)

        monkeypatch.setattr(supervisor.cluster_view, 'change', record)
        for index, (action, group) in enumerate(
            (
                (lambda: supervisor.label('group', 'mua', [0]), 'mua'),
                (supervisor.undo, 'good'),
                (supervisor.redo, 'mua'),
            )
        ):
            metrics.clear()
            builtin_calls.clear()
            action()
            supervisor.block()
            assert supervisor.cluster_view._model.row_by_id(0)['group'] == group
            assert supervisor.cluster_view._model.row_by_id(0)['quality_metric'] == group
            assert supervisor.selection.snapshot() == selected
            if index == 0:
                assert set(metrics) == set(supervisor.clustering.cluster_ids)
                assert set(builtin_calls) == {0}
                assert set(patches[-1]) == {'id', 'quality_metric'}
            assert view.get_data(0) is data
            assert (
                tuple(
                    artist for ax in view.canvas.iter_ax() for artist in (*ax.lines, *ax.patches)
                )
                == artists
            )
        assert events.count('bounded') == 3
        assert events.count('cluster') == 3
    finally:
        unconnect(on_state, on_cluster)
        gui.close()
        controller.close()


def test_metadata_refresh_preserves_table_filter_sort_and_selection(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    supervisor = controller.supervisor
    try:
        table = supervisor.cluster_view
        table.sort_by('id', 'desc')
        table.filter("group == 'good'")
        supervisor.select([0])
        supervisor.block()
        assert 0 in table.get_ids()
        assert supervisor.selected == [0]
        supervisor.label('group', 'mua', [0])
        supervisor.block()
        assert table._model.row_by_id(0)['group'] == 'mua'
        assert 0 not in table.get_ids()
        assert supervisor.selected == table.get_selected_ids()
        assert 0 not in supervisor.selected
        assert table._current_sort == ('id', 'desc')
        supervisor.undo()
        supervisor.block()
        assert table._model.row_by_id(0)['group'] == 'good'
        assert 0 in table.get_ids()
        assert supervisor.selected == [0]
        assert table.get_selected_ids() == [0]
        assert table._current_sort == ('id', 'desc')
        supervisor.redo()
        supervisor.block()
        assert 0 not in table.get_ids()
        assert 0 not in supervisor.selected
    finally:
        gui.close()
        controller.close()


def test_membership_undo_and_new_summary_still_refresh_stability(qtbot, tmp_path, monkeypatch):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    supervisor = controller.supervisor
    try:
        supervisor.select([0])
        supervisor.block()
        view = gui.get_view('UnitStabilityView')
        view.on_select([0])
        old = view.get_data(0)
        calls = []
        plot = view.plot

        def measured(*args, **kwargs):
            calls.append(True)
            return plot(*args, **kwargs)

        monkeypatch.setattr(view, 'plot', measured)
        supervisor.merge([0, 1], to=100)
        supervisor.block()
        assert calls
        assert int(view.get_data(100).counts.sum()) == 7
        calls.clear()
        supervisor.undo()
        supervisor.block()
        assert calls
        assert view.get_data(0) is not old
        np.testing.assert_array_equal(view.get_data(0).counts, old.counts)
        calls.clear()
        current = view.get_data(0)
        emit('bounded_summary_changed', supervisor)
        assert calls
        assert view.get_data(0) is not current
    finally:
        gui.close()
        controller.close()


def test_native_membership_failure_still_refreshes_unchanged_revision(
    qtbot, tmp_path, monkeypatch
):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    supervisor = controller.supervisor
    view = gui.get_view('UnitStabilityView')
    try:
        supervisor.select([0])
        supervisor.block()
        view.on_select([0])
        before = view.get_data(0)
        revision = supervisor.session.membership_revision
        calls = []
        plot = view.plot

        def measured(*args, **kwargs):
            calls.append(True)
            return plot(*args, **kwargs)

        @connect(event='cluster', sender=supervisor)
        def reject(sender, up):
            if not up.description.startswith('metadata_'):
                raise RuntimeError('reject native membership')

        monkeypatch.setattr(view, 'plot', measured)
        with pytest.raises(RuntimeError, match='reject native membership'):
            supervisor.merge([0, 1], to=100)
        assert calls
        assert supervisor.session.membership_revision is revision
        assert tuple(view.cluster_ids) == (0,)
        np.testing.assert_array_equal(view.get_data(0).counts, before.counts)
    finally:
        if 'reject' in locals():
            unconnect(reject)
        gui.close()
        controller.close()


def test_array_metric_label_undo_redo_compares_without_ambiguous_truth(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    supervisor = controller.supervisor
    supervisor.cluster_metrics['array_metric'] = lambda cluster: np.array(
        [cluster, len(supervisor.session.events), np.nan]
    )
    supervisor.columns.append('array_metric')
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    try:
        supervisor.select([0])
        supervisor.block()
        for action, group, events in (
            (lambda: supervisor.label('group', 'mua', [0]), 'mua', 1),
            (supervisor.undo, 'good', 0),
            (supervisor.redo, 'mua', 1),
        ):
            action()
            supervisor.block()
            row = supervisor.cluster_view._model.row_by_id(0)
            assert row['group'] == group
            np.testing.assert_array_equal(row['array_metric'], [0, events, np.nan])
            other = supervisor.cluster_view._model.row_by_id(2)
            np.testing.assert_array_equal(other['array_metric'], [2, events, np.nan])
    finally:
        gui.close()
        controller.close()


@pytest.mark.parametrize(
    'previous, current, expected',
    [
        (np.array([1, np.nan]), np.array([1, np.nan]), True),
        (np.array(['a', 'b']), np.array(['a', 'b']), True),
        (np.array(['a', 'b']), np.array(['a', 'c']), False),
        (np.array([1, 2]), np.array([1]), False),
        ('good', 'good', True),
        ('good', 'mua', False),
    ],
)
def test_table_metric_value_equality(previous, current, expected):
    from phy.apps.large_recording.supervisor import _table_values_equal

    assert _table_values_equal(previous, current) is expected


@pytest.mark.parametrize('name', ['good_count', 'ch'])
def test_cross_row_metric_and_replaced_builtin_refresh_every_live_row(qtbot, tmp_path, name):
    _, controller = make_controller(tmp_path)
    supervisor = controller.supervisor
    calls = []

    def count_good(cluster):
        calls.append(cluster)
        return sum(
            supervisor.cluster_meta.get('group', unit) == 'good'
            for unit in supervisor.clustering.cluster_ids
        )

    supervisor.cluster_metrics[name] = count_good
    if name not in supervisor.columns:
        supervisor.columns.append(name)
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    try:
        assert supervisor.cluster_view._model.row_by_id(2)[name] == 2
        for action, expected in (
            (lambda: supervisor.label('group', 'mua', [0]), 1),
            (supervisor.undo, 2),
            (supervisor.redo, 1),
        ):
            calls.clear()
            action()
            supervisor.block()
            assert set(calls) == set(supervisor.clustering.cluster_ids)
            for unit in supervisor.clustering.cluster_ids:
                assert supervisor.cluster_view._model.row_by_id(unit)[name] == expected
    finally:
        gui.close()
        controller.close()
