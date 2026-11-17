"""Exact lineage and read-only parent inspection regressions."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from phylib.utils import Bunch, emit

from phy.apps._parent_review import ParentLineage, ParentProviders, _ParentLegend
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

    from phy.gui.qt import QLabel, Qt

    controller = _mock_controller(tempdir, MyControllerFull)
    gui = controller.create_gui(do_prompt_save=False)
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
            legend = window.findChild(QLabel, 'parent-review-legend')
            assert window._toolbar.isVisible()
            assert return_button.isVisible() and return_button.isEnabled()
            assert return_button.text() == 'Return to merged unit'
            assert return_button.visibleRegion().contains(return_button.rect().center())
            assert window.rect().contains(
                return_button.mapTo(window, return_button.rect().center())
            )
            assert legend.isVisible() and legend.wordWrap()
            assert legend.height() >= legend.heightForWidth(legend.width())
            assert window._toolbar.height() < min(180, window.height() / 4)
            assert f'{target} ← 0' in legend.text() and f'{target} ← 1' in legend.text()
            docks = {type(v): v.dock.geometry() for v in inspection.views}
            assert docks[CorrelogramView].left() < docks[WaveformView].left()
            assert docks[CorrelogramView].top() < docks[AmplitudeView].top()
            assert docks[WaveformView].top() < docks[FeatureView].top()
            assert docks[AmplitudeView].left() < docks[FeatureView].left()
            assert docks[CorrelogramView].width() >= docks[WaveformView].width()
            assert all(rect.width() >= 300 and rect.height() >= 180 for rect in docks.values())
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


@pytest.mark.parametrize('count', [2, 7, 16])
def test_parent_legend_uses_only_the_height_of_its_wrapped_text(qtbot, count):
    from phy.gui.qt import QMainWindow, Qt, QToolBar

    text = (
        ' &nbsp; | &nbsp; '.join(
            f'<span style="color:rgb(0,150,255)">705 ← {parent} '
            f'(2057 retained spikes; 612536 full spikes; 1.71 Hz)</span>'
            for parent in range(count)
        )
        + '<br>Window evidence only; no full-recording correlogram rate baseline.'
    )
    window = QMainWindow()
    qtbot.addWidget(window)
    toolbar = QToolBar(window)
    window.addToolBar(toolbar)
    toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
    action = toolbar.addAction('Return to merged unit')
    legend = _ParentLegend(text)
    toolbar.addWidget(legend)
    window.resize(1880, 1120)
    window.show()
    for width in (1880, 1100):
        window.resize(width, 1120)
        qtbot.wait(30)
        assert legend.isVisible()
        assert legend.width() > width / 2
        needed = legend.heightForWidth(legend.width())
        assert needed <= legend.height() <= needed + 8
        assert toolbar.height() <= legend.height() + 16
        assert toolbar.height() < window.height() / 3
        if count == 2:
            assert toolbar.height() < 180
        button = toolbar.widgetForAction(action)
        assert button.isVisible() and button.visibleRegion().contains(button.rect().center())
        assert all(f'705 ← {parent} ' in legend.text() for parent in range(count))


def test_bounded_saved_nested_and_split_parents_use_journal_not_templates(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    path = controller.dir_path
    config = controller.config_dir
    try:
        supervisor = controller.supervisor
        first = supervisor.merge([0, 1]).added[0]
        target = supervisor.merge([first, 2]).added[0]
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
