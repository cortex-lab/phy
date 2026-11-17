"""Exact lineage and read-only parent inspection regressions."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from phylib.utils import Bunch, emit

from phy.apps._parent_review import ParentLineage, ParentProviders
from phy.apps.large_recording.tests.test_adapter import make_controller
from phy.apps.tests.test_base import MyControllerFull, _mock_controller
from phy.cluster.clustering import Clustering
from phy.cluster.views import AmplitudeView, CorrelogramView, FeatureView, WaveformView


def lineage_controller(path, assignments):
    return SimpleNamespace(
        dir_path=path,
        supervisor=SimpleNamespace(clustering=Clustering(np.asarray(assignments))),
        model=SimpleNamespace(spike_times=np.arange(len(assignments))),
    )


def test_exact_nested_split_undo_redo_membership(tmp_path):
    controller = lineage_controller(tmp_path, [0, 1, 0, 1, 2, 2, 3, 3])
    lineage = ParentLineage(controller)
    clustering = controller.supervisor.clustering
    try:
        first = clustering.merge([0, 1]).added[0]
        nested = clustering.merge([first, 2]).added[0]
        groups = lineage.parents(nested)
        assert groups[2].tolist() == [4, 5]
        assert groups[first].tolist() == [0, 1, 2, 3]
        clustering.undo()
        assert set(lineage.parents(first)) == {0, 1}
        clustering.redo()
        assert set(lineage.parents(nested)) == {first, 2}
        split = clustering.split([0, 4])
        children = split.added
        before = {int(c): clustering.spikes_per_cluster[c].copy() for c in children}
        merged = clustering.merge(children).added[0]
        for parent, ids in lineage.parents(merged).items():
            np.testing.assert_array_equal(ids, before[parent])
            assert not ids.flags.writeable
        with pytest.raises(ValueError, match='unavailable'):
            lineage.parents(3)
    finally:
        lineage.close()


def test_lineage_save_reopen_and_stale_assignment_rejection(tmp_path):
    controller = lineage_controller(tmp_path, [0, 0, 1, 1, 2, 2])
    lineage = ParentLineage(controller)
    target = controller.supervisor.clustering.merge([0, 1]).added[0]
    assignments = controller.supervisor.clustering.spike_clusters.copy()
    emit('curation_saved', controller.supervisor)
    lineage.close()
    reopened = lineage_controller(tmp_path, assignments)
    loaded = ParentLineage(reopened)
    assert loaded.parents(target)[0].tolist() == [0, 1]
    loaded.close()
    changed = assignments.copy()
    changed[[0, 5]] = changed[[5, 0]]
    stale = ParentLineage(lineage_controller(tmp_path, changed))
    try:
        with pytest.raises(ValueError, match='unavailable'):
            stale.parents(target)
    finally:
        stale.close()


def test_lineage_rejects_membership_that_does_not_partition_live_unit(tmp_path):
    controller = lineage_controller(tmp_path, [0, 0, 1, 1])
    lineage = ParentLineage(controller)
    try:
        target = controller.supervisor.clustering.merge([0, 1]).added[0]
        lineage.records[target][0] = np.array([0])
        with pytest.raises(ValueError, match='partition'):
            lineage.parents(target)
    finally:
        lineage.close()


def test_lineage_rejects_changed_source_rows_and_corrupt_archives(tmp_path):
    controller = lineage_controller(tmp_path, [0, 0, 1, 1])
    lineage = ParentLineage(controller)
    target = controller.supervisor.clustering.merge([0, 1]).added[0]
    emit('curation_saved', controller.supervisor)
    lineage.close()
    controller.model.spike_times[[0, 1]] = controller.model.spike_times[[1, 0]]
    changed = ParentLineage(controller)
    try:
        with pytest.raises(ValueError, match='unavailable'):
            changed.parents(target)
    finally:
        changed.close()
    (tmp_path / '_phy_merge_parents.npz').write_bytes(b'PK\x03\x04broken')
    corrupt = ParentLineage(controller)
    try:
        with pytest.raises(ValueError, match='unavailable'):
            corrupt.parents(target)
    finally:
        corrupt.close()


def test_providers_use_exact_spikes_without_changing_controller(tempdir):
    controller = _mock_controller(tempdir, MyControllerFull)
    try:
        c = controller.supervisor.clustering
        original = {p: c.spikes_per_cluster[p].copy() for p in (0, 1)}
        target = c.merge([0, 1]).added[0]
        groups = controller.parent_inspection.lineage.parents(target)
        providers = ParentProviders(controller, target, groups)
        for parent, values in original.items():
            np.testing.assert_array_equal(groups[parent], values)
            features = providers.features(parent)
            assert set(features.spike_ids) <= set(values)
            waveforms = providers.waveforms(parent)
            assert set(waveforms.spike_ids) <= set(values)
        amplitudes = providers.amplitudes([None, 0, 1])
        for parent, bunch in zip((0, 1), amplitudes[1:]):
            assert set(bunch.spike_ids) <= set(original[parent])
            np.testing.assert_array_equal(
                bunch.amplitudes, controller.model.amplitudes[bunch.spike_ids]
            )
        values = providers.correlograms([0, 1], 0.001, 0.01)
        assert values.shape[0:2] == (2, 2)
        assert not np.any(c.spike_clusters[original[0]] != target)
    finally:
        controller.close()


def test_review_is_read_only_and_closes_on_selection_and_mutation(qtbot, tempdir, monkeypatch):
    from phylib.utils.event import _EVENT

    from phy.gui.qt import Qt

    controller = _mock_controller(tempdir, MyControllerFull)
    gui = controller.create_gui(do_prompt_save=False)
    gui.resize(1100, 800)
    gui.show()
    inspection = controller.parent_inspection
    supervisor = controller.supervisor
    try:
        target = supervisor.merge([0, 1]).added[0]
        supervisor.select([target])
        supervisor.block()
        supervisor.save()
        saved_lineage = inspection.lineage.path.read_bytes()
        before = supervisor.clustering.spike_clusters.copy()
        metadata = supervisor.get_labels('group').copy()
        dirty = supervisor.is_dirty()
        history = supervisor.clustering._undo_stack._index
        original = controller._get_waveforms
        selection_settings = dict(controller.selection)
        for _ in range(2):
            callbacks = list(_EVENT._callbacks)
            window = inspection.show(target)
            qtbot.wait(30)
            assert len(inspection.views) == 4
            assert {type(v) for v in inspection.views} == {
                WaveformView,
                FeatureView,
                AmplitudeView,
                CorrelogramView,
            }
            for view in inspection.views:
                assert view.cluster_ids == [0, 1]
                assert not hasattr(view, 'actions')
                view.on_mouse_click(Bunch(key='Control', button='Left', pos=(0, 0)))
            return_action = window.file_actions.get('return_to_merged_unit')
            return_button = window._toolbar.widgetForAction(return_action)
            assert window._toolbar.isVisible()
            assert return_button.isVisible() and return_button.isEnabled()
            assert return_button.text() == 'Back to merged unit'
            assert return_button.visibleRegion().contains(return_button.rect().center())
            assert window.rect().contains(
                return_button.mapTo(window, return_button.rect().center())
            )
            assert inspection.host_gui is gui
            assert not window.isWindow()
            assert inspection.parent_table.rowCount() == 2
            assert inspection.parent_table.isVisible()
            assert window._toolbar.height() < min(100, window.height() / 4)
            docks = {type(v): v.dock.geometry() for v in inspection.views}
            assert docks[CorrelogramView].left() < docks[WaveformView].left()
            assert docks[CorrelogramView].top() < docks[AmplitudeView].top()
            assert docks[WaveformView].top() < docks[FeatureView].top()
            assert docks[AmplitudeView].left() < docks[FeatureView].left()
            assert docks[CorrelogramView].width() >= docks[WaveformView].width()
            assert all(rect.width() >= 200 and rect.height() >= 150 for rect in docks.values())
            np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before)
            assert supervisor.get_labels('group') == metadata
            assert supervisor.is_dirty() == dirty
            assert supervisor.clustering._undo_stack._index == history
            assert controller._get_waveforms == original
            assert dict(controller.selection) == selection_settings
            assert supervisor.selected == [target]
            with monkeypatch.context() as patcher:

                def forbidden_wait(*args):
                    pytest.fail('Parent review closure must not pump curation events')

                patcher.setattr('phy.gui.gui._wait', forbidden_wait)
                # A QAction trigger or QTest click on a hidden widget would also
                # work. Require visible, reachable geometry before real input.
                qtbot.mouseClick(return_button, Qt.LeftButton)
            assert inspection.window is None
            assert inspection.lineage.path.read_bytes() == saved_lineage
            assert _EVENT._callbacks == callbacks
        inspection.show(target)
        supervisor.select([2])
        supervisor.block()
        assert inspection.window is None
        inspection.show(target)
        supervisor.merge([target, 2])
        assert inspection.window is None
    finally:
        inspection.close_review()
        gui.close()
        controller.close()


def test_bounded_saved_nested_and_split_parents_use_journal_not_templates(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    path = controller.dir_path
    config = controller.config_dir
    try:
        supervisor = controller.supervisor
        first = supervisor.merge([0, 1]).added[0]
        target = supervisor.merge([first, 2]).added[0]
        expected_colors = dict(controller.parent_inspection.lineage.colors(target))
        assert expected_colors == {first: 0, 2: 1}
        before = supervisor.clustering.spike_clusters.copy()
        groups = controller.parent_inspection.lineage.parents(target)
        assert set(groups) == {first, 2}
        assert set(groups[first]) == {0, 1, 3, 4, 6, 7, 9}
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before)
        supervisor.save()
    finally:
        controller.close()
    from phy.apps.template.gui import TemplateController

    reopened = TemplateController(dir_path=path, config_dir=config, plugins=[])
    try:
        groups = reopened.parent_inspection.lineage.parents(target)
        assert set(groups) == {first, 2}
        assert dict(reopened.parent_inspection.lineage.colors(target)) == expected_colors
        assert len(groups[first]) == 7
        assert reopened.parent_inspection.lineage.full_counts(target) == {first: 7, 2: 3}
        provider = ParentProviders(reopened, target, groups)
        result = provider.correlograms(list(groups), 0.001, 0.01)
        assert result.shape[:2] == (2, 2)
        assert not (path / '_phy_merge_parents.npz').exists()
    finally:
        reopened.close()


def test_bounded_predicate_split_merge_reload_and_read_only_window(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    path = controller.dir_path
    config = controller.config_dir
    try:
        supervisor = controller.supervisor
        split = supervisor.split_source_amplitude(0, 4.5)
        children = split.added
        parents = [children[0], 1]
        expected = {p: supervisor.clustering.spikes_per_cluster[p].copy() for p in parents}
        target = supervisor.merge(parents).added[0]
        supervisor.save()
    finally:
        controller.close()
    from phy.apps.template.gui import TemplateController

    reopened = TemplateController(dir_path=path, config_dir=config, plugins=[])
    gui = reopened.create_gui(do_prompt_save=False)
    try:
        supervisor = reopened.supervisor
        supervisor.select([target])
        supervisor.block()
        session = reopened.bounded_session
        assignments, events = session.assignments.copy(), deepcopy(session.events)
        dirty = supervisor.is_dirty()
        groups = reopened.parent_inspection.lineage.parents(target)
        for parent in parents:
            np.testing.assert_array_equal(groups[parent], expected[parent])
        assert reopened.parent_inspection.lineage.full_counts(target)[children[0]] is None
        reopened.parent_inspection.show(target)
        assert reopened.parent_inspection.window is not None
        reopened.parent_inspection.close_review()
        np.testing.assert_array_equal(session.assignments, assignments)
        assert session.events == events
        assert supervisor.is_dirty() == dirty
    finally:
        gui.close()
        reopened.close()


def test_premerge_palette_slots_survive_save_reload_undo_and_redo(tmp_path):
    # Sparse/high authoritative slots are meaningful: rendering wraps the palette,
    # but display metadata must retain the exact slot, not silently renumber it.
    controller = lineage_controller(tmp_path, [9, 15, 10, 13, 9, 15, 10, 13])
    expected = {9: 0, 15: 300, 10: 2, 13: 3}
    controller.supervisor.selection_color_indices = expected
    lineage = ParentLineage(controller)
    try:
        emit('before_merge', controller.supervisor, (9, 15, 10, 13))
        target = controller.supervisor.clustering.merge([9, 15, 10, 13]).added[0]
        assert dict(lineage.colors(target)) == expected
        assert list(lineage.parents(target)) == [9, 10, 13, 15]
        controller.supervisor.clustering.undo()
        controller.supervisor.clustering.redo()
        assert dict(lineage.colors(target)) == expected
        emit('curation_saved', controller.supervisor)
    finally:
        lineage.close()
    reopened = ParentLineage(controller)
    try:
        assert dict(reopened.colors(target)) == expected
        with pytest.raises(TypeError):
            reopened.colors(target)[9] = 4
    finally:
        reopened.close()
    # Optional corrupt hints must not invalidate the exact scientific partition.
    path = tmp_path / '_phy_merge_parents.npz'
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays['color_indices'] = np.asarray('{"1":{"9":-1}}')
    np.savez(path, **arrays)
    fallback = ParentLineage(controller)
    try:
        assert set(fallback.parents(target)) == set(expected)
        assert dict(fallback.colors(target)) == {9: 0, 10: 1, 13: 2, 15: 3}
    finally:
        fallback.close()


def test_color_observer_does_not_limit_large_merges(tmp_path):
    from phy.utils.color import colormaps

    count = len(colormaps.default) + 1
    controller = lineage_controller(tmp_path, np.arange(count))
    lineage = ParentLineage(controller)
    try:
        emit('before_merge', controller.supervisor, tuple(range(count)))
        target = controller.supervisor.clustering.merge(list(range(count))).added[0]
        assert len(lineage.parents(target)) == count
        assert target not in lineage.color_records
    finally:
        lineage.close()


def test_real_unsorted_merge_palette_and_exact_workspace_restoration(qtbot, tempdir, monkeypatch):
    from phy.gui.qt import QAction, QApplication, QDockWidget, Qt

    controller = _mock_controller(tempdir, MyControllerFull)
    gui = controller.create_gui(do_prompt_save=False)
    gui.resize(1100, 800)
    gui.show()
    supervisor = controller.supervisor
    inspection = controller.parent_inspection
    try:
        supervisor.select([9, 15, 10, 13])
        supervisor.block()
        supervisor.toggle_merge_mode()
        colors = dict(supervisor.selection_color_indices)
        assert colors == {9: 0, 15: 1, 10: 2, 13: 3}
        target = supervisor.merge().added[0]
        supervisor.select([target])
        supervisor.block()
        assert dict(inspection.lineage.colors(target)) == colors
        supervisor.save()
        # Preserve deliberate hidden/disabled states, including a custom action.
        ordinary = gui.list_views(WaveformView)[0]
        ordinary.dock.hide()
        custom = QAction('Custom disabled action', gui)
        custom.setEnabled(False)
        gui.addAction(custom)
        supervisor.cluster_view.table_view.setFocus()
        qtbot.wait(20)
        state = bytes(gui.saveState())
        gui.restoreState(gui.saveState())  # Normalize Qt's cached pre-show central size.
        qtbot.wait(20)
        state = bytes(gui.saveState())
        docks = {
            d: not d.isHidden() for d in gui.findChildren(QDockWidget) if d.parentWidget() is gui
        }
        actions = {a: a.isEnabled() for a in gui.findChildren(QAction)}
        original_focus = QApplication.focusWidget()
        original_views = tuple(gui.views)
        assignments = supervisor.clustering.spike_clusters.copy()
        history = supervisor.clustering._undo_stack._index
        saved = inspection.lineage.path.read_bytes()
        inspection.show(target)
        qtbot.wait(30)
        assert len(inspection.views) == 4
        assert tuple(gui.views) == original_views
        assert [inspection.parent_table.item(i, 1).text() for i in range(4)] == [
            '9',
            '15',
            '10',
            '13',
        ]
        assert inspection.parent_table.isColumnHidden(3)
        assert all(dict(v._cluster_color_index_by_id) == colors for v in inspection.views)
        assert all(
            not action.qaction.isEnabled() for action in supervisor.actions._actions_dict.values()
        )
        # Attempts to re-enable actions (for example snippets.mode_off) remain blocked.
        supervisor.actions.get('merge').setEnabled(True)
        assert not supervisor.actions.get('merge').isEnabled()
        for key, modifiers in (
            (Qt.Key_G, Qt.NoModifier),
            (Qt.Key_K, Qt.NoModifier),
            (Qt.Key_Z, Qt.ControlModifier),
            (Qt.Key_S, Qt.ControlModifier),
        ):
            qtbot.keyClick(inspection.return_button, key, modifier=modifiers)
        assert inspection.window is not None
        assert supervisor.clustering._undo_stack._index == history
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, assignments)
        assert inspection.lineage.path.read_bytes() == saved
        # Esc returns without any generic GUI wait/save path.
        with monkeypatch.context() as patcher:
            patcher.setattr('phy.gui.gui._wait', lambda *a: pytest.fail('nested event pumping'))
            qtbot.keyClick(inspection.return_button, Qt.Key_Escape)
        assert inspection.window is None
        assert bytes(gui.saveState()) == state
        assert all(d.isHidden() != visible for d, visible in docks.items())
        assert all(a.isEnabled() == enabled for a, enabled in actions.items())
        assert QApplication.focusWidget() is original_focus
        # A partial child-GUI constructor must not leak global event callbacks.
        from phylib.utils.event import _EVENT

        callbacks = list(_EVENT._callbacks)
        with monkeypatch.context() as patcher:

            def fail_table(*args):
                raise RuntimeError('injected table construction failure')

            patcher.setattr('phy.apps._parent_workspace.QTableWidget', fail_table)
            with pytest.raises(RuntimeError, match='injected table'):
                inspection.show(target)
        assert inspection.window is None
        assert _EVENT._callbacks == callbacks
        assert bytes(gui.saveState()) == state
        # A provider failure after workspace construction must leave no hidden docks.
        with monkeypatch.context() as patcher:
            patcher.setattr(
                ParentProviders,
                'waveform_ids',
                lambda *a: (_ for _ in ()).throw(ValueError('injected provider failure')),
            )
            with pytest.raises(ValueError, match='injected provider'):
                inspection.show(target)
        assert inspection.window is None
        assert bytes(gui.saveState()) == state
        assert all(a.isEnabled() == enabled for a, enabled in actions.items())
        # Roll back even if activation fails after hiding/adding real host docks.
        from phy.apps._parent_workspace import ParentWorkspace

        activate = ParentWorkspace._activate
        with monkeypatch.context() as patcher:

            def fail_after_activation(workspace):
                activate(workspace)
                raise RuntimeError('injected activation failure')

            patcher.setattr(ParentWorkspace, '_activate', fail_after_activation)
            with pytest.raises(RuntimeError, match='injected activation'):
                inspection.show(target)
        assert inspection.window is None
        assert bytes(gui.saveState()) == state
        assert all(a.isEnabled() == enabled for a, enabled in actions.items())
        # Main-close geometry is saved only after the temporary workspace is gone.
        inspection.show(target)
        seen = []
        original_save = gui.save_geometry_state

        def save_original_layout():
            seen.append((inspection.window, bytes(gui.saveState())))
            original_save()

        monkeypatch.setattr(gui, 'save_geometry_state', save_original_layout)
        gui.close()
        assert seen == [(None, state)]
    finally:
        inspection.close_review()
        gui.close()
        controller.close()


@pytest.mark.parametrize('count', [7, 16])
def test_parent_table_keeps_all_rows_readable_and_scrollable(qtbot, tempdir, count):
    controller = _mock_controller(tempdir, MyControllerFull)
    gui = controller.create_gui(do_prompt_save=False)
    gui.resize(1100, 800)
    gui.show()
    try:
        supervisor = controller.supervisor
        target = supervisor.merge(list(range(count))).added[0]
        supervisor.select([target])
        supervisor.block()
        inspection = controller.parent_inspection
        inspection.show(target)
        qtbot.wait(30)
        table = inspection.parent_table
        assert table.rowCount() == count
        assert inspection.window._toolbar.height() < 100
        assert table.width() <= 280
        for row in range(count):
            item = table.item(row, 1)
            table.scrollToItem(item)
            qtbot.wait(1)
            assert table.visualItemRect(item).intersects(table.viewport().rect())
            assert table.rowHeight(row) >= table.fontMetrics().height()
            assert item.text() == str(row)
        assert all(v.dock.width() >= 200 and v.dock.height() >= 150 for v in inspection.views)
    finally:
        controller.parent_inspection.close_review()
        gui.close()
        controller.close()


def test_bounded_color_hints_are_disposable_and_save_errors_are_isolated(
    qtbot, tmp_path, monkeypatch
):
    import json

    from phy.apps.template.gui import TemplateController

    _, controller = make_controller(tmp_path)
    path, config = controller.dir_path, controller.config_dir
    try:
        supervisor = controller.supervisor
        first = supervisor.merge([0, 1]).added[0]
        target = supervisor.merge([first, 2]).added[0]
        lineage = controller.parent_inspection.lineage
        expected = {first: 0, 2: 1}
        assert dict(lineage.colors(target)) == expected
        supervisor.save()
        hint = lineage._bounded_color_path()
        assert hint.is_file() and hint.parent == config
        value = json.loads(hint.read_text())
        # Refuse valid colors attached to a different saved semantic history.
        value['binding'] = 'stale'
        hint.write_text(json.dumps(value))
    finally:
        controller.close()
    reopened = TemplateController(dir_path=path, config_dir=config, plugins=[])
    try:
        lineage = reopened.parent_inspection.lineage
        assert set(lineage.parents(target)) == {first, 2}
        assert dict(lineage.colors(target)) == {2: 0, first: 1}
        # A committed scientific save must succeed even if this optional cache cannot.
        original_replace = __import__('os').replace

        def fail_optional(source, destination):
            if destination == lineage._bounded_color_path():
                raise OSError('injected optional cache write error')
            return original_replace(source, destination)

        with monkeypatch.context() as patcher:
            patcher.setattr('phy.apps._parent_review.os.replace', fail_optional)
            reopened.supervisor.save()
        assert reopened.bounded_session.checkpoint_path.is_dir()
        assert set(lineage.parents(target)) == {first, 2}
        hint.write_text('{broken optional cache')
    finally:
        reopened.close()
    final = TemplateController(dir_path=path, config_dir=config, plugins=[])
    try:
        assert set(final.parent_inspection.lineage.parents(target)) == {first, 2}
        assert dict(final.parent_inspection.lineage.colors(target)) == {2: 0, first: 1}
    finally:
        final.close()


def test_parent_review_can_reopen_after_canceled_dirty_main_close(qtbot, tempdir, monkeypatch):
    from phy.gui.qt import Qt

    controller = _mock_controller(tempdir, MyControllerFull)
    gui = controller.create_gui()
    gui.resize(1100, 800)
    gui.show()
    inspection = controller.parent_inspection
    supervisor = controller.supervisor
    try:
        target = supervisor.merge([0, 1]).added[0]
        supervisor.select([target])
        supervisor.block()
        assert supervisor.is_dirty()
        assignments = supervisor.clustering.spike_clusters.copy()
        history = supervisor.clustering._undo_stack._index
        metadata = supervisor.get_labels('group').copy()
        inspection.show(target)
        qtbot.wait(20)
        with monkeypatch.context() as patcher:
            patcher.setattr('phy.apps.base._prompt_save', lambda: 'cancel')
            assert not gui.close()
        assert gui.isVisible() and not gui._closed
        assert inspection.window is None
        assert inspection._gui is gui
        inspection.show_selected()
        qtbot.wait(20)
        assert inspection.host_gui is gui
        assert inspection.return_button.isVisible()
        qtbot.mouseClick(inspection.return_button, Qt.LeftButton)
        assert inspection.window is None
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, assignments)
        assert supervisor.clustering._undo_stack._index == history
        assert supervisor.get_labels('group') == metadata
        assert supervisor.is_dirty()
    finally:
        inspection.close_review()
        gui.close()
        controller.close()
