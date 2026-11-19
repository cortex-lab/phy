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
