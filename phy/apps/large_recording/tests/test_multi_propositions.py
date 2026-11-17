"""Combined suggestion review is exact and journaled only on explicit merge."""

import numpy as np

from ..adapter import BoundedTemplateController
from ..replay import apply_curation
from .test_adapter import make_controller


def test_combined_suggestions_checkpoint_and_replay(tmp_path, qtbot):
    mapping = {
        'format_version': '2',
        'unit_ids': [0, 1, 2, 99],
        'merges': [{'unit_ids': [0, 1]}, {'unit_ids': [1, 2]}],
    }
    source_params, controller = make_controller(tmp_path, propositions=mapping)
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    supervisor = controller.supervisor
    keys = tuple(p.key for p in supervisor.merge_propositions.catalog.propositions)
    original = controller.bounded_session.assignments.copy()
    producer_before = (controller.dir_path / 'curation.json').read_bytes()
    try:
        supervisor._activate_merge_propositions(supervisor.merge_propositions_view, keys)
        assert supervisor.selected == [0, 1, 2]
        assert not controller.bounded_session.events
        np.testing.assert_array_equal(controller.bounded_session.assignments, original)
        target = supervisor.merge().added[0]
        assert controller.bounded_session.events[-1]['parents'] == [0, 1, 2]
        assert len(supervisor.merge_propositions.catalog.reviews) == 2
        supervisor.undo()
        np.testing.assert_array_equal(controller.bounded_session.assignments, original)
        assert not supervisor.merge_propositions.catalog.reviews
        assert supervisor.selection.state.merge.proposition_ids == keys
        supervisor.redo()
        supervisor.save()
        assert (controller.dir_path / 'curation.json').read_bytes() == producer_before
        package = controller.dir_path
    finally:
        gui.close()
        controller.close()
    reopened = BoundedTemplateController(
        dir_path=package, config_dir=tmp_path / 'reopened', plugins=[]
    )
    try:
        assert len(reopened.supervisor.merge_propositions.catalog.reviews) == 2
        assert all(
            r.result_unit_id == target
            for r in reopened.supervisor.merge_propositions.catalog.reviews.values()
        )
        assert reopened.bounded_session.events[-1]['parents'] == [0, 1, 2]
    finally:
        reopened.close()

    output = apply_curation(source_params, package, tmp_path / 'curated')
    source_assignments = np.load(source_params.parent / 'spike_clusters.npy')
    expected = source_assignments.copy()
    expected[np.isin(expected, [0, 1, 2])] = target
    np.testing.assert_array_equal(np.load(output / 'spike_clusters.npy'), expected)
