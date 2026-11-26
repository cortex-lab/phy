"""Batch navigation must not change curation targets or saved assignments."""

import sys

import numpy as np

from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.tests.test_adapter import make_controller
from phy.apps.template.gui import TemplateController
from phy.gui.qt import Qt


def test_good_then_batch_at_end_saves_reopens_and_replays(qtbot, tmp_path):
    params, controller = make_controller(tmp_path)
    path = controller.dir_path
    source_bytes = (params.parent / 'spike_clusters.npy').read_bytes()
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    gui.show()
    try:
        supervisor = controller.supervisor
        target = supervisor.merge([0, 2]).added[0]
        expected = supervisor.clustering.spike_clusters.copy()
        supervisor.cluster_view.sort_by('id', 'asc')
        supervisor.select([target])
        supervisor.block()
        modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier
        # No event-loop wait between the two shortcuts and Save.
        qtbot.keyClick(gui, Qt.Key_G, Qt.AltModifier)
        qtbot.keyClick(gui, Qt.Key_Space, modifier)
        supervisor.save()
        supervisor.block()
        assert supervisor.selection.state.reference_id is None
        assert supervisor.similarity_view.get_ids() == []
        assert supervisor.cluster_meta.get('group', target) == 'good'
        assert not supervisor.is_dirty()
        assert not supervisor.task_logger._processing
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, expected)
    finally:
        gui.close()
        controller.close()
    reopened = TemplateController(dir_path=path, config_dir=tmp_path / 'config', plugins=[])
    try:
        np.testing.assert_array_equal(reopened.bounded_session.assignments, expected)
        assert reopened.supervisor.cluster_meta.get('group', target) == 'good'
        replay = apply_curation(params, path, tmp_path / 'replay', batch_size=2)
        np.testing.assert_array_equal(np.load(replay / 'spike_clusters.npy'), expected)
        assert f'{target}\tgood\n' in (replay / 'cluster_group.tsv').read_text()
        assert (params.parent / 'spike_clusters.npy').read_bytes() == source_bytes
    finally:
        reopened.close()


def test_reordered_blue_proposition_preserves_colors_history_and_saved_membership(qtbot, tmp_path):
    propositions = {
        'format_version': '2',
        'unit_ids': [0, 1, 2, 99],
        'merges': [{'unit_ids': [0, 2]}],
    }
    params, controller = make_controller(tmp_path, propositions=propositions)
    path = controller.dir_path
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    try:
        supervisor = controller.supervisor
        key = supervisor.merge_propositions.catalog.propositions[0].key
        supervisor._activate_merge_proposition(supervisor.merge_propositions_view, key)
        view = supervisor.merge_view
        view.emit_cluster_drop(view, (0,), 2)
        staged = supervisor.selection.snapshot()
        assert staged.merge_ids == (2, 0)
        assert staged.reference_id == 0
        assert dict(staged.color_indices) == {0: 0, 2: 1}
        supervisor.actions.merge()
        supervisor.block()
        target = supervisor.selected_clusters[0]
        expected = supervisor.clustering.spike_clusters.copy()
        supervisor.undo()
        supervisor.block()
        assert supervisor.selection.state == staged
        supervisor.redo()
        supervisor.block()
        supervisor.save()
        assert dict(controller.parent_inspection.lineage.colors(target)) == {0: 0, 2: 1}
    finally:
        gui.close()
        controller.close()
    reopened = TemplateController(dir_path=path, config_dir=tmp_path / 'config', plugins=[])
    try:
        np.testing.assert_array_equal(reopened.bounded_session.assignments, expected)
        assert dict(reopened.parent_inspection.lineage.colors(target)) == {0: 0, 2: 1}
        assert set(reopened.parent_inspection.lineage.parents(target)) == {0, 2}
        replay = apply_curation(params, path, tmp_path / 'replay', batch_size=2)
        np.testing.assert_array_equal(np.load(replay / 'spike_clusters.npy'), expected)
        review = reopened.supervisor.merge_propositions.catalog.reviews[key]
        assert review.result_unit_id == target
        assert set(review.applied_unit_ids) == {0, 2}
    finally:
        reopened.close()
