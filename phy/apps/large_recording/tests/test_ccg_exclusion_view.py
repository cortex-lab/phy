"""Sampled-only controls, stable evidence, and visualization/curation separation."""

import numpy as np
import pytest

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.tests.test_adapter import make_gui_sorting
from phy.apps.large_recording.tests.test_ccg_exclusion import _all_pairs
from phy.apps.large_recording.views import SampledCorrelogramView
from phy.apps.template.gui import TemplateController
from phy.cluster.views import CorrelogramView
from phy.gui import GUI
from phy.gui.qt import QApplication, QLineEdit, QPoint, Qt, QWidget


def _provider(ids, bin_size, window_size, **kwargs):
    return np.zeros((len(ids), len(ids), 5), dtype=np.int64), np.zeros((len(ids), len(ids)))


def test_legacy_provider_and_ordinary_view_keep_their_contract(qtbot, tmp_path):
    calls = []

    def legacy(ids, bin_size, window_size):
        calls.append((ids, bin_size, window_size))
        return _provider(ids, bin_size, window_size)

    gui = GUI(config_dir=tmp_path / 'config')
    qtbot.addWidget(gui)
    view = SampledCorrelogramView(correlograms=legacy, sample_rate=30000)
    try:
        view.attach(gui)
        view.cluster_ids = [0, 1]
        view.get_clusters_data()
        assert len(calls) == 1
        assert view.actions.get('toggle_auto_exclusion') is None
        assert view.actions.get('toggle_cross_exclusion') is None
        assert 'toggle_auto_exclusion' not in CorrelogramView.default_shortcuts
        assert 'toggle_cross_exclusion' not in CorrelogramView.default_shortcuts
    finally:
        gui.close()


@pytest.mark.parametrize('invalid', [-0.1, float('nan'), float('inf'), 'invalid'])
def test_exclusion_width_rejects_invalid_values_without_changing_settings(invalid):
    view = SampledCorrelogramView(
        correlograms=_provider, exclusion_correlograms=_provider, sample_rate=30000
    )
    try:
        with pytest.raises(ValueError):
            view.set_exclusion_zone(invalid)
        assert view.exclusion_ms == 0.3
        assert not view.exclude_auto and not view.exclude_cross
        view.set_exclusion_zone(0)
        assert view.exclusion_ms == 0
    finally:
        view.canvas.close()


@pytest.mark.parametrize('auto_key', [Qt.Key_D, Qt.Key_X])
def test_shortcuts_are_dock_scoped_and_independent(qtbot, tmp_path, auto_key):
    gui = GUI(config_dir=tmp_path / 'config')
    qtbot.addWidget(gui)
    view = SampledCorrelogramView(
        correlograms=_provider,
        exclusion_correlograms=_provider,
        sample_rate=30000,
        shortcuts={'toggle_auto_exclusion': 'x'} if auto_key == Qt.Key_X else None,
    )
    other = QLineEdit()
    try:
        view.attach(gui)
        gui.add_view(other)
        with qtbot.waitExposed(gui):
            gui.show()
        gui.activateWindow()
        qtbot.waitUntil(gui.isActiveWindow)
        qtbot.mouseClick(view.canvas, Qt.LeftButton, pos=QPoint(20, 20))
        qtbot.waitUntil(
            view.canvas.hasFocus if isinstance(view.canvas, QWidget) else view.canvas.isActive
        )
        qtbot.keyClick(view.canvas, auto_key)
        qtbot.waitUntil(lambda: view.exclude_auto)
        assert not view.exclude_cross
        assert view.actions.get('toggle_auto_exclusion').isChecked()
        qtbot.keyClick(view.canvas, Qt.Key_D, modifier=Qt.ShiftModifier)
        qtbot.waitUntil(lambda: view.exclude_cross)
        assert view.exclude_auto
        assert 'EXCL auto+cross ±0.3 ms' in view.dock.status
        other.setFocus()
        qtbot.waitUntil(other.hasFocus)
        qtbot.keyClicks(other, 'dD')
        assert other.text() == 'dD'
        assert view.exclude_auto and view.exclude_cross
        second = SampledCorrelogramView(
            correlograms=_provider, exclusion_correlograms=_provider, sample_rate=30000
        )
        second.attach(gui)
        assert not second.actions.get('toggle_auto_exclusion').shortcut().toString()
        assert not second.actions.get('toggle_cross_exclusion').shortcut().toString()
    finally:
        gui.close()


def _make_close_spike_controller(tmp_path):
    params = make_gui_sorting(tmp_path / 'source')
    np.save(
        params.parent / 'spike_times.npy',
        np.array([0, 3, 12, 60, 63, 75, 90, 93, 105, 120, 123, 135], dtype=np.int64),
    )
    output = prepare_large(
        params, tmp_path / 'bounded', window_seconds=20, stratum_seconds=20, max_spikes=100
    )
    return params, TemplateController(
        dir_path=output, config_dir=tmp_path / 'config', plugins=[], enable_threading=False
    )


@pytest.mark.parametrize('auto_key', [Qt.Key_D, Qt.Key_X])
def test_exclusion_keys_in_full_bounded_gui_change_counts_without_curation(
    qtbot, tmp_path, auto_key
):
    _, controller = _make_close_spike_controller(tmp_path)
    gui = controller.create_gui(do_prompt_save=False)
    qtbot.addWidget(gui)
    view = gui.get_view('SampledCorrelogramView')
    if auto_key == Qt.Key_X:
        view.actions.get('toggle_auto_exclusion').setShortcut('X')
    assignments = controller.bounded_session.assignments.copy()
    try:
        with qtbot.waitExposed(gui):
            gui.show()
        gui.activateWindow()
        qtbot.waitUntil(gui.isActiveWindow)
        controller.supervisor.select([0, 1, 2])
        controller.supervisor.block()
        view.set_exclusion_zone(2)
        original, _ = view._get_correlograms_data()
        view.dock.raise_()
        qtbot.mouseClick(view.canvas, Qt.LeftButton, pos=QPoint(20, 20))
        qtbot.waitUntil(
            view.canvas.hasFocus if isinstance(view.canvas, QWidget) else view.canvas.isActive
        )
        if not isinstance(view.canvas, QWidget):
            view.canvas.requestActivate()
            qtbot.waitUntil(lambda: QApplication.focusWindow() == view.canvas)
        qtbot.keyClick(view.canvas, auto_key)
        qtbot.waitUntil(lambda: view.exclude_auto)
        auto, _ = view._get_correlograms_data()
        assert auto[0, 0].sum() < original[0, 0].sum()
        np.testing.assert_array_equal(auto[0, 1], original[0, 1])
        qtbot.keyClick(view.canvas, Qt.Key_D, modifier=Qt.ShiftModifier)
        qtbot.waitUntil(lambda: view.exclude_cross)
        both, _ = view._get_correlograms_data()
        assert both[0, 1].sum() < original[0, 1].sum()
        np.testing.assert_array_equal(both[0, 0], auto[0, 0])
        qtbot.keyClick(view.canvas, auto_key)
        qtbot.keyClick(view.canvas, Qt.Key_D, modifier=Qt.ShiftModifier)
        qtbot.waitUntil(lambda: not view.exclude_auto and not view.exclude_cross)
        np.testing.assert_array_equal(view._get_correlograms_data()[0], original)
        np.testing.assert_array_equal(controller.bounded_session.assignments, assignments)
        assert not controller.bounded_session.events
        assert not controller.supervisor.is_dirty()
    finally:
        gui.close()
        controller.close()


def test_view_exclusion_preserves_data_history_and_settings_across_reopen(qtbot, tmp_path):
    params, controller = _make_close_spike_controller(tmp_path)
    gui = controller.create_gui(default_views=['SampledCorrelogramView'], do_prompt_save=False)
    qtbot.addWidget(gui)
    directory = controller.dir_path
    view = gui.get_view('SampledCorrelogramView')
    session = controller.bounded_session
    immutable = {
        path: path.read_bytes()
        for folder in (params.parent, directory)
        for path in folder.glob('*.npy')
    }
    assignments = session.assignments.copy()
    history = controller.supervisor._global_history.current_position
    evidence_ids = controller.model.spike_samples.copy()
    try:
        controller.supervisor.select([0, 1, 2])
        controller.supervisor.block()
        view.set_bin(0.5)
        original, baseline = view._get_correlograms_data()
        center = original.shape[2] // 2
        view.toggle_cross_exclusion(True)
        filtered, adjusted = view._get_correlograms_data()
        assert original[0, 1, center] > 0
        assert filtered[0, 1, center] == 0  # 0.1 ms pairs excluded.
        assert filtered[0, 2, center] > 0  # 0.4 ms pairs in that same bin retained.
        np.testing.assert_array_equal(filtered[0, 0], original[0, 0])
        np.testing.assert_array_equal(adjusted, baseline)
        view.toggle_cross_exclusion(False)
        np.testing.assert_array_equal(view._get_correlograms_data()[0], original)
        view.set_exclusion_zone(2)
        view.toggle_auto_exclusion(True)
        assert view._get_correlograms_data()[0][0, 0].sum() < original[0, 0].sum()
        view.toggle_cross_exclusion(True)
        assert '±2 ms' in view.status
        assert not session.events
        assert controller.supervisor._global_history.current_position == history
        assert not controller.supervisor.is_dirty()
        np.testing.assert_array_equal(session.assignments, assignments)
        np.testing.assert_array_equal(controller.model.spike_samples, evidence_ids)
        assert all(path.read_bytes() == data for path, data in immutable.items())
        # Large exclusion widths can remove every pair; the empty histogram still renders.
        view.set_exclusion_zone(1000)
        assert view._get_correlograms_data()[0].sum() == 0
        view.plot()
        view.set_exclusion_zone(2)
    finally:
        gui.close()
        controller.close()

    reopened = TemplateController(
        dir_path=directory, config_dir=tmp_path / 'config', plugins=[], enable_threading=False
    )
    reopened_gui = reopened.create_gui(
        default_views=['SampledCorrelogramView'], do_prompt_save=False
    )
    qtbot.addWidget(reopened_gui)
    try:
        restored = reopened_gui.get_view('SampledCorrelogramView')
        assert restored.exclusion_ms == 2
        assert restored.exclude_auto and restored.exclude_cross
        assert restored.actions.get('toggle_auto_exclusion').isChecked()
        assert restored.actions.get('toggle_cross_exclusion').isChecked()
        assert not reopened.bounded_session.events
        assert not reopened.supervisor.is_dirty()
    finally:
        reopened_gui.close()
        reopened.close()


def test_exclusion_uses_live_memberships_after_merge_split_and_undo(tmp_path):
    _, controller = _make_close_spike_controller(tmp_path)
    view = controller.create_sampled_correlogram_view()
    supervisor = controller.supervisor
    samples = controller.model.spike_samples.copy()
    view._set_bin_window(bin_size=0.0005, window_size=0.003)
    view.exclude_auto = view.exclude_cross = True

    def checked_counts(ids):
        view.cluster_ids = ids
        actual, _ = view._get_correlograms_data()
        expected = _all_pairs(
            samples,
            supervisor.clustering.spike_clusters,
            controller.evidence_windows,
            ids,
            15,
            3,
            9,
            True,
            True,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(controller.model.spike_samples, samples)
        return actual

    try:
        before = checked_counts([0, 1])
        merged = supervisor.merge([0, 1]).added
        after_merge = checked_counts(merged)
        children = supervisor.split_source_amplitude(merged[0], 4.5).added
        checked_counts(children)
        supervisor.undo()
        np.testing.assert_array_equal(checked_counts(merged), after_merge)
        supervisor.undo()
        np.testing.assert_array_equal(checked_counts([0, 1]), before)
    finally:
        view.canvas.close()
        controller.close()
