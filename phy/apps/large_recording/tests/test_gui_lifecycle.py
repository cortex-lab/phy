"""GUI transaction, proposition review, and pending-metric regressions."""

import json

import numpy as np
import pytest
from phylib.utils import connect, unconnect

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.tests.test_adapter import make_controller, make_gui_sorting
from phy.apps.template.gui import TemplateController


def test_elapsed_producer_windows_open_and_navigate_in_gui(qtbot, tmp_path):
    params = make_gui_sorting(tmp_path / 'source')
    intervals = [[0, 210000], [300000, 600000]]
    contract = {'kind': 'elapsed_ticks', 'ticks_per_second': 30000.0}
    params.write_text(
        params.read_text() + f'time_base = {contract!r}\nvalid_intervals = {intervals!r}\n'
    )
    np.save(params.parent / '_phy_spikes_subset.window_bounds.npy', np.array(intervals, 'int64'))
    (params.parent / '_phy_spikes_subset.manifest.json').write_text(
        json.dumps(
            {
                'format': 'phy_spikes_subset',
                'format_version': 1,
                'recording': {'window_bounds_coordinates': 'elapsed_ticks'},
            }
        )
    )
    output = prepare_large(params, tmp_path / 'bounded', window_policy='producer', bin_seconds=5)
    controller = TemplateController(
        dir_path=output, config_dir=tmp_path / 'config', plugins=[], enable_threading=False
    )
    gui = None
    try:
        gui = controller.create_gui(do_prompt_save=False)
        controller.supervisor.select([0, 1])
        controller.supervisor.block()
        controller.inspect_evidence_window(1)
        assert controller.model.sample_rate == 30000.0
        assert controller.model.traces is None
        assert controller.get_mean_firing_rate(0) == pytest.approx(4 / 17)
        view = gui.get_view('EvidenceAmplitudeView')
        assert view.duration == 10
        assert view.window_label.text() == ('Evidence window 1 | Recording time: 10.000–20.000 s')
    finally:
        if gui is not None:
            gui.close()
        controller.close()


def test_window_time_labels_follow_navigation(qtbot, tmp_path):
    _, controller = make_controller(tmp_path, sparse=True)
    gui = controller.create_gui(default_views=['EvidenceAmplitudeView'], do_prompt_save=False)
    try:
        view = gui.get_view('EvidenceAmplitudeView')
        assert view.time_axis_label.text() == 'Recording time (seconds)'
        assert view.amplitude_scope == 'overview'
        initial = view.window_label.text()
        controller.inspect_evidence_window(1)
        start, end = controller.current_evidence_window / controller.model.sample_rate
        assert view.window_label.text() == (
            f'Evidence window 1 | Recording time: {start:.3f}–{end:.3f} s'
        )
        assert view.window_label.text() != initial
        assert view.duration == end - start
    finally:
        gui.close()
        controller.close()


def test_gui_observer_mutation_rolls_back_views_selection_and_journal(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    supervisor = controller.supervisor
    once = []

    try:
        supervisor.select([0])
        supervisor.block()
        assignments = supervisor.clustering.spike_clusters.copy()
        events = list(controller.bounded_session.events)
        selection = supervisor.selection.snapshot()
        cluster_ids = supervisor.cluster_view.get_ids()
        similarity_ids = supervisor.similarity_view.get_ids()
        histories = tuple(
            (history._index, len(history._history))
            for history in (
                supervisor.clustering._undo_stack,
                supervisor.cluster_meta._undo_stack,
                supervisor._global_history,
                supervisor.session_history,
            )
        )
        dirty = supervisor.is_dirty()

        @connect(event='cluster', sender=supervisor.clustering)
        def mutate(sender, up):
            if not once:
                once.append(True)
                sender.assign([0])

        with pytest.raises((ValueError, KeyError)):
            supervisor.merge([0, 1])
        supervisor.block()

        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, assignments)
        assert controller.bounded_session.events == events
        assert supervisor.selection.snapshot() == selection
        assert supervisor.cluster_view.get_ids() == cluster_ids
        assert supervisor.similarity_view.get_ids() == similarity_ids
        assert (
            tuple(
                (history._index, len(history._history))
                for history in (
                    supervisor.clustering._undo_stack,
                    supervisor.cluster_meta._undo_stack,
                    supervisor._global_history,
                    supervisor.session_history,
                )
            )
            == histories
        )
        assert supervisor.is_dirty() == dirty
    finally:
        if 'mutate' in locals():
            unconnect(mutate)
        gui.close()
        controller.close()


def test_gui_proposition_review_undo_redo_and_checkpoint_reopen(qtbot, tmp_path):
    params = make_gui_sorting(tmp_path / 'source')
    proposal = {
        'format_version': '2',
        'unit_ids': [0, 1, 2, 99],
        'merges': [{'unit_ids': [0, 1]}],
    }
    (params.parent / 'curation.json').write_text(json.dumps(proposal), encoding='utf-8')
    bounded = prepare_large(
        params,
        tmp_path / 'bounded',
        window_seconds=20,
        stratum_seconds=20,
        bin_seconds=5,
        batch_size=2,
    )
    config = tmp_path / 'config'
    controller = TemplateController(
        dir_path=bounded, config_dir=config, plugins=[], enable_threading=False
    )
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    supervisor = controller.supervisor
    proposition = supervisor.merge_propositions.catalog.propositions[0]
    key = proposition.key

    try:
        supervisor._review_merge_proposition(supervisor.merge_propositions_view, key)
        assert supervisor.selection.state.is_merge_mode
        assert supervisor.selected_merge == [0, 1]

        merged = supervisor.merge()
        supervisor.block()
        assert merged.added == [100]
        assert supervisor.merge_propositions.catalog.status_for(key).value == 'accepted'
        assert controller.bounded_session.events[-1]['target'] == 100

        supervisor.undo()
        supervisor.block()
        assert supervisor.merge_propositions.catalog.status_for(key).value == 'pending'
        assert controller.bounded_session.events == []

        supervisor.redo()
        supervisor.block()
        assert supervisor.merge_propositions.catalog.status_for(key).value == 'accepted'
        assert controller.bounded_session.events[-1]['target'] == 100
        supervisor.save()
    finally:
        gui.close()
        controller.close()

    reopened = TemplateController(
        dir_path=bounded, config_dir=config, plugins=[], enable_threading=False
    )
    try:
        key = reopened.supervisor.merge_propositions.catalog.propositions[0].key
        assert reopened.supervisor.merge_propositions.catalog.status_for(key).value == 'accepted'
        assert reopened.bounded_session.events[-1]['target'] == 100
    finally:
        reopened.close()


def test_default_bounded_views_handle_threshold_children_with_pending_metrics(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(do_prompt_save=False)
    try:
        supervisor = controller.supervisor
        update = supervisor.split_source_amplitude(0, 4)
        supervisor.block()
        assert set(update.added) == {100, 101}
        for cluster_id in update.added:
            row = supervisor.cluster_view._model.row_by_id(cluster_id)
            assert row['fr'] is None
            assert row['full_count'] is None
            assert row['evidence'] == 'pending replay'
    finally:
        gui.close()
        controller.close()


def test_merge_redo_then_amplitude_split_saves_and_reopens(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    bounded = controller.dir_path
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    supervisor = controller.supervisor
    try:
        supervisor.merge([0, 1])
        supervisor.block()
        supervisor.undo()
        supervisor.block()
        supervisor.redo()
        supervisor.block()
        split = supervisor.split_source_amplitude(2, 6.0)
        supervisor.block()
        assignments = supervisor.clustering.spike_clusters.copy()
        supervisor.save()
        children = list(split.added)
    finally:
        gui.close()
        controller.close()
    reopened = TemplateController(
        dir_path=bounded, config_dir=tmp_path / 'config', plugins=[], enable_threading=False
    )
    try:
        np.testing.assert_array_equal(reopened.bounded_session.assignments, assignments)
        assert set(reopened.bounded_session.events[-1]['children']) == set(children)
        assert len(reopened.bounded_session.events) == 2
    finally:
        reopened.close()
