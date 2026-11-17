"""Exact lineage and read-only parent inspection regressions."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from phylib.utils import emit

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


def test_review_is_read_only_and_closes_on_selection_and_mutation(qtbot, tempdir):
    from phylib.utils.event import _EVENT

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
        for _ in range(2):
            callbacks = list(_EVENT._callbacks)
            window = inspection.show(target)
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
            np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before)
            assert supervisor.get_labels('group') == metadata
            assert supervisor.is_dirty() == dirty
            assert supervisor.clustering._undo_stack._index == history
            assert controller._get_waveforms == original
            assert supervisor.selected == [target]
            window.file_actions.get('return_to_merged_unit').trigger()
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
