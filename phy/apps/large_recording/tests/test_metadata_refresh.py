"""Metadata edits update table semantics without rebuilding scientific evidence."""

import numpy as np
import pytest
from phylib.utils import connect, emit, unconnect

from phy.apps.large_recording.tests.test_adapter import make_controller


def test_label_undo_redo_keep_scientific_artists_and_refresh_only_changed_rows(
    qtbot, tmp_path, monkeypatch
):
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
            action()
            supervisor.block()
            assert supervisor.cluster_view._model.row_by_id(0)['group'] == group
            assert supervisor.cluster_view._model.row_by_id(0)['quality_metric'] == group
            assert supervisor.selection.snapshot() == selected
            if index == 0:
                assert set(metrics) == {0}
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
