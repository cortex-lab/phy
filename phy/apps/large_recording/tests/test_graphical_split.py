"""Graphical threshold previews commit full-source semantic commands."""

import numpy as np
import pytest
from phylib.utils import connect, emit, unconnect

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.replay import apply_curation
from phy.apps.large_recording.tests.test_adapter import make_controller, make_gui_sorting
from phy.apps.template.gui import TemplateController
from phy.gui.qt import Qt
from phy.plot.tests import mouse_click, mouse_drag
from phy.plot.transform import NDC, Range


def drag_threshold(qtbot, view, threshold):
    qtbot.wait(50)
    width, height = view.canvas.get_size()
    ndc = Range(view.data_bounds, NDC).apply([[0, threshold]])[0]
    screen = (ndc + np.asarray(view.canvas.panzoom.pan)) * view.canvas.panzoom._zoom_aspect()
    pixel = ((screen[0] + 1) * width / 2, (1 - screen[1]) * height / 2)
    mouse_drag(
        qtbot, view.canvas, (pixel[0] - 10, pixel[1]), pixel, button='right', modifiers=('Alt',)
    )
    assert view.split_threshold == pytest.approx(threshold, abs=0.03)


def test_drag_k_semantic_commit_waveform_preview_history_save_replay(qtbot, tmp_path):
    params, controller = make_controller(tmp_path)
    gui = controller.create_gui(
        default_views=['EvidenceAmplitudeView', 'WaveformView'], do_prompt_save=False
    )
    supervisor = controller.supervisor
    requests = []

    @connect(event='request_split', sender=supervisor)
    def on_request(sender):
        requests.append(True)
        return np.array([0], dtype=np.int64)

    try:
        gui.show()
        supervisor.select([0])
        supervisor.block()
        view = gui.get_view('EvidenceAmplitudeView')
        waveform = gui.get_view('WaveformView')
        waveform.waveforms_type = 'waveforms'
        waveform.plot()
        drag_threshold(qtbot, view, 4.5)
        threshold = view.split_threshold
        drawn = waveform._displayed_bunchs[0].spike_ids
        amplitudes = controller.bounded_session.attributes['amplitudes.npy']
        np.testing.assert_array_equal(
            waveform._highlighted_spike_ids, drawn[amplitudes[drawn] < threshold]
        )
        assert waveform._highlighted_spike_color == view.split_preview_color
        assert supervisor.actions.get('split').isEnabled()
        assert view.on_request_split(supervisor).size == 0
        before = supervisor.clustering.spike_clusters.copy()
        view.canvas.setFocus()
        qtbot.keyClick(view.canvas, Qt.Key_K)
        supervisor.block()
        assert not requests
        assert controller.bounded_session.events[-1]['operation'] == 'threshold_split'
        predicate = controller.bounded_session.events[-1]['predicate']
        assert predicate['threshold'] == threshold
        after = supervisor.clustering.spike_clusters.copy()
        assert view.split_threshold is None
        assert not supervisor.actions.get('split').isEnabled()
        supervisor.undo()
        supervisor.block()
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before)
        supervisor.redo()
        supervisor.block()
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, after)
        supervisor.save()
        bounded = controller.dir_path
    finally:
        unconnect(on_request)
        gui.close()
        controller.close()

    reopened = TemplateController(dir_path=bounded, config_dir=tmp_path / 'config', plugins=[])
    try:
        np.testing.assert_array_equal(reopened.supervisor.clustering.spike_clusters, after)
        result = apply_curation(params, bounded, tmp_path / 'replay', batch_size=2)
        source = np.load(params.parent / 'spike_clusters.npy')
        values = np.load(params.parent / 'amplitudes.npy')
        children = reopened.bounded_session.events[-1]['children']
        expected = source.copy()
        expected[(source == 0) & (values < threshold)] = children[1]
        expected[(source == 0) & ~(values < threshold)] = children[0]
        np.testing.assert_array_equal(np.load(result / 'spike_clusters.npy'), expected)
    finally:
        reopened.close()


def test_overview_original_clock_all_windows_budget_and_window_toggle(qtbot, tmp_path):
    params = make_gui_sorting(tmp_path / 'source')
    bounded = prepare_large(
        params, tmp_path / 'bounded', window_seconds=3, stratum_seconds=5, bin_seconds=5
    )
    controller = TemplateController(dir_path=bounded, config_dir=tmp_path / 'config', plugins=[])
    gui = controller.create_gui(default_views=['EvidenceAmplitudeView'], do_prompt_save=False)
    try:
        gui.show()
        controller.supervisor.select([0])
        controller.supervisor.block()
        view = gui.get_view('EvidenceAmplitudeView')
        assert view.amplitude_scope == 'overview'
        data = view.get_clusters_data(load_all=True)[0]
        assert len(data.spike_ids) > len(controller._window_ids(0))
        np.testing.assert_allclose(
            data.spike_times, controller.model.spike_samples[data.spike_ids] / 30000
        )
        controller.n_spikes_amplitudes = 2
        controller.n_spikes_amplitudes_total = 2
        assert len(view.get_clusters_data()[-1].spike_ids) <= 2
        drag_threshold(qtbot, view, 4.5)
        controller.inspect_evidence_window(0)
        assert view.split_threshold is None
        assert view.amplitude_scope == 'window'
        local = view.get_clusters_data(load_all=True)[0]
        np.testing.assert_array_equal(local.spike_ids, controller._window_ids(0))
        view.toggle_amplitude_scope()
        assert view.amplitude_scope == 'overview'
        assert view.state['amplitude_scope'] == 'overview'
    finally:
        gui.close()
        controller.close()


def test_invalid_preview_and_row_splits_preserve_state(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=['EvidenceAmplitudeView'], do_prompt_save=False)
    supervisor = controller.supervisor
    try:
        gui.show()
        supervisor.select([0])
        supervisor.block()
        view = gui.get_view('EvidenceAmplitudeView')
        drag_threshold(qtbot, view, 4.5)
        before = supervisor.clustering.spike_clusters.copy()
        with pytest.raises(ValueError, match='cannot be replayed'):
            supervisor.split([0])
        assert view.split_threshold is not None
        view.split_threshold = -1
        with pytest.raises(ValueError, match='Both sides'):
            supervisor.split()
        assert view.split_threshold == -1
        supervisor.actions.get('split').trigger()
        supervisor.block()
        assert view.dock.status == 'Both sides of the threshold must have evidence spikes.'
        assert view.split_threshold == -1
        assert supervisor.actions.get('split').isEnabled()
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before)
        assert controller.bounded_session.events == []
        view.clear_split_selection()
        view.canvas.lasso.add((-0.5, -0.5))
        view.canvas.lasso.add((0.5, -0.5))
        view.canvas.lasso.add((0, 0.5))
        assert view.on_request_split(supervisor).size == 0
        with pytest.raises(ValueError, match='Exactly one'):
            supervisor.split()
        assert not supervisor.actions.get('split').isEnabled()
    finally:
        gui.close()
        controller.close()


def test_preview_invalidated_by_context_and_hidden_dock(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(
        default_views=['EvidenceAmplitudeView', 'WaveformView'], do_prompt_save=False
    )
    supervisor = controller.supervisor
    try:
        gui.show()
        supervisor.select([0])
        supervisor.block()
        view = gui.get_view('EvidenceAmplitudeView')
        waveform = gui.get_view('WaveformView')
        drag_threshold(qtbot, view, 4.5)
        waveform.plot()
        assert view.split_threshold is None
        for event in ('selected_channel_changed', 'selected_feature_changed'):
            drag_threshold(qtbot, view, 4.5)
            emit(event, waveform)
            assert view.split_threshold is None
            assert not len(waveform._highlighted_spike_ids)
        drag_threshold(qtbot, view, 4.5)
        waveform.next_waveforms_type()
        assert view.split_threshold is None
        waveform.waveforms_type = 'waveforms'
        waveform.plot()
        assert not len(waveform._highlighted_spike_ids)
        drag_threshold(qtbot, view, 4.5)
        mouse_click(qtbot, view.canvas, (20, 20), button='right', modifiers=('Control',))
        assert view.split_threshold is None
        drag_threshold(qtbot, view, 4.5)
        view.actions.get('clear_amplitude_split_threshold').trigger()
        assert view.split_threshold is None
        drag_threshold(qtbot, view, 4.5)
        view.dock.hide()
        assert view.split_threshold is None
        assert not supervisor.actions.get('split').isEnabled()
        view.dock.show()
        drag_threshold(qtbot, view, 4.5)
        supervisor.select([0, 1])
        supervisor.block()
        assert view.split_threshold is None
        assert not view._can_set_split_threshold()
        supervisor.select([0])
        supervisor.block()
        drag_threshold(qtbot, view, 4.5)
        supervisor.toggle_merge_mode()
        supervisor.block()
        assert view.split_threshold is None
        assert not view._can_set_split_threshold()
    finally:
        gui.close()
        controller.close()


def test_commit_scope_other_gui_ambiguous_and_closed_previews(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=['EvidenceAmplitudeView'], do_prompt_save=False)
    supervisor = controller.supervisor
    other = None
    try:
        gui.show()
        supervisor.select([0])
        supervisor.block()
        first = gui.get_view('EvidenceAmplitudeView')
        second = gui.create_and_add_view('EvidenceAmplitudeView')
        supervisor.select([0])
        supervisor.block()
        # Simulate contradictory preview state without the exclusive activation
        # event, as a plugin or restored stale callback could create it.
        first.split_threshold = 3
        second.split_threshold = 4
        with pytest.raises(ValueError, match='Exactly one'):
            supervisor.split()
        second.clear_split_selection()
        # A foreign view answering the global legacy event cannot intercept K.
        other = controller.create_evidence_amplitude_view()
        other.on_select([0])
        other.split_threshold = 6
        other.gui = object()
        foreign_requests = []

        @connect(event='request_split')
        def foreign_request(sender):
            foreign_requests.append(True)
            return np.array([1], dtype=np.int64)

        try:
            supervisor.split()
            supervisor.block()
            assert not foreign_requests
            assert controller.bounded_session.events[-1]['predicate']['threshold'] == 3
        finally:
            unconnect(foreign_request)
        supervisor.undo()
        supervisor.block()
        supervisor.select([0])
        supervisor.block()
        drag_threshold(qtbot, first, 4.5)
        # Closing the originating dock invalidates its pending predicate.
        first.close()
        assert first.split_threshold is None
        with pytest.raises(ValueError, match='Exactly one'):
            supervisor.split()
    finally:
        if other is not None:
            other.canvas.close()
        gui.close()
        controller.close()


def test_failed_semantic_observer_preserves_preview_and_assignments(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=['EvidenceAmplitudeView'], do_prompt_save=False)
    supervisor = controller.supervisor
    try:
        gui.show()
        supervisor.select([0])
        supervisor.block()
        view = gui.get_view('EvidenceAmplitudeView')
        drag_threshold(qtbot, view, 4.5)
        threshold = view.split_threshold
        before = supervisor.clustering.spike_clusters.copy()

        @connect(event='cluster', sender=supervisor.clustering)
        def fail(sender, up):
            raise RuntimeError('observer failed')

        try:
            with pytest.raises(RuntimeError, match='observer failed'):
                supervisor.split()
            supervisor.block()
            np.testing.assert_array_equal(supervisor.clustering.spike_clusters, before)
            assert controller.bounded_session.events == []
            assert supervisor.selected == [0]
            assert view.split_threshold == threshold
            assert supervisor.actions.get('split').isEnabled()
        finally:
            unconnect(fail)
    finally:
        gui.close()
        controller.close()


def test_pending_reference_selection_invalidates_old_split_preview(qtbot, tmp_path):
    from timeit import default_timer

    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(
        view_count={'EvidenceAmplitudeView': 1},
        default_views=['EvidenceAmplitudeView'],
        do_prompt_save=False,
    )
    supervisor = controller.supervisor
    try:
        gui.show()
        supervisor.select([0])
        supervisor.block()
        view = gui.get_view('EvidenceAmplitudeView')
        drag_threshold(qtbot, view, 4.5)
        assert supervisor._graphical_split_views() == [view]
        assignments = supervisor.clustering.spike_clusters.copy()
        supervisor.cluster_view.debouncer.delay = 60_000
        supervisor.cluster_view.debouncer._last_submission_time = default_timer()
        supervisor.cluster_view.select([1])
        assert supervisor.selected == [0]
        assert supervisor.cluster_view.get_selected_ids() == [1]

        with pytest.raises(ValueError, match='threshold preview'):
            supervisor.split()
        assert supervisor.selected == [1]
        assert view.split_threshold is None
        assert not controller.bounded_session.events
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, assignments)
    finally:
        gui.close()
        controller.close()
