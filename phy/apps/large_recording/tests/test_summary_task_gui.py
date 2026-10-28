"""Nonmodal store task workflow and live handover without reopening."""

from threading import Event

import pytest

from phy.apps.large_recording import summary_task_gui
from phy.apps.large_recording.summary_store import StoreEstimate, summary_store_reference
from phy.apps.large_recording.summary_tasks import StoreInspection
from phy.apps.large_recording.tests.test_adapter import make_controller
from phy.gui.qt import QTimer
from phy.gui.tasks import TaskProgress


@pytest.fixture
def opened(tmp_path):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    actions = controller._summary_store_actions
    actions._checked = True  # Automatic prompt gets its own explicit test below.
    try:
        yield controller, gui, actions
    finally:
        gui.close()
        controller.close()


def test_build_keeps_gui_responsive_and_adopts_store_after_curation(
    qtbot, opened, tmp_path, monkeypatch
):
    from phy.apps.large_recording.live_summaries import LiveSummaryEngine

    controller, gui, actions = opened
    original = summary_task_gui.prepare_summary_store
    started, release, built, deliver = Event(), Event(), Event(), Event()

    def controlled(request, context):
        started.set()
        assert release.wait(5)
        result = original(request, context)
        built.set()
        assert deliver.wait(5)
        return result

    monkeypatch.setattr(summary_task_gui, 'prepare_summary_store', controlled)
    heartbeat = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: heartbeat.append(True))
    timer.start()
    try:
        actions.build(tmp_path / 'store')
        qtbot.waitUntil(started.is_set)
        children = controller.supervisor.split_source_amplitude(0, 4).added
        controller.supervisor.undo()
        controller.supervisor.redo()
        qtbot.waitUntil(lambda: len(heartbeat) >= 5)
        assert controller.get_full_count(children[0]) is None
        assert 'waiting' in controller.get_summary_status(children[0])
        release.set()
        qtbot.waitUntil(built.is_set)

        def forbid_source(*args, **kwargs):
            raise AssertionError('GUI handover rescanned the original source')

        monkeypatch.setattr(LiveSummaryEngine, '_open', forbid_source)
        deliver.set()
        qtbot.waitUntil(lambda: summary_store_reference(controller.dir_path) is not None)
        qtbot.waitUntil(lambda: controller.get_full_count(children[0]) == 2)
        assert not controller._live_summaries._suspended
        assert actions.manager.state == 'succeeded'
        assert controller.supervisor.is_dirty()  # Building must not save curation.
        assert 'ready' in actions.widget.label.text().lower()
    finally:
        timer.stop()
        release.set()
        deliver.set()
        qtbot.waitUntil(lambda: not actions.manager.busy)


def test_accepted_close_cancels_old_task_without_attaching_to_reopened_gui(
    qtbot, opened, tmp_path, monkeypatch
):
    controller, gui, old = opened
    started, release = Event(), Event()

    def blocked(request, context):
        started.set()
        assert release.wait(5)
        context.check_cancel()

    monkeypatch.setattr(summary_task_gui, 'prepare_summary_store', blocked)
    old.build(tmp_path / 'store')
    qtbot.waitUntil(started.is_set)
    gui.close()
    reopened = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    controller._summary_store_actions._checked = True
    try:
        release.set()
        qtbot.waitUntil(lambda: not old.manager.busy)
        assert summary_store_reference(controller.dir_path) is None
        assert controller._summary_store_actions is not old
        assert not controller._live_summaries._suspended
    finally:
        release.set()
        reopened.close()


def test_prompt_is_single_and_suppression_is_dataset_local(qtbot, opened):
    controller, gui, actions = opened
    actions.build_dialog()
    qtbot.waitUntil(lambda: actions.dialog is not None)
    dialog = actions.dialog
    actions.build_dialog()
    assert actions.dialog is dialog
    dialog.suppress.setChecked(True)
    dialog._continue()
    assert gui.state['SummaryStore']['suppress_prompt']
    assert not actions.manager.busy
    assert not controller._live_summaries._suspended


def test_reference_read_error_is_recoverable(qtbot, opened, monkeypatch):
    _, _, actions = opened

    def unreadable(*args, **kwargs):
        raise OSError('permission denied')

    monkeypatch.setattr(summary_task_gui, 'summary_store_reference', unreadable)
    actions.build_dialog()
    assert 'permission denied' in actions.widget.label.text()
    assert not actions.manager.busy
    assert not actions.service._suspended


def test_prompt_budget_and_async_preflight(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(summary_task_gui, '_free_space', lambda path: 1024 * 1073741824)
    estimate = StoreEstimate(100 * 1073741824, 100 * 1073741824, 100, 6, str(tmp_path / 'source'))
    prompt = summary_task_gui.StorePrompt(
        StoreInspection(estimate, False, True), tmp_path / 'store'
    )
    try:
        qtbot.waitUntil(lambda: 'budget' in prompt.problem.text())
        assert not prompt.build_button.isEnabled()
        prompt.budget.setValue(128)
        qtbot.waitUntil(prompt.build_button.isEnabled)
        prompt.destination.setText(str(tmp_path))
        qtbot.waitUntil(lambda: 'new directory' in prompt.problem.text())
        assert not prompt.build_button.isEnabled()
    finally:
        prompt.close()


def test_cancel_text_survives_late_progress(qtbot, opened):
    _, _, actions = opened
    actions.manager.state = 'cancelling'
    actions.widget.on_progress(TaskProgress('Building', 1, 100, 'rows'))
    assert actions.widget.estimate.text() == 'Cancelling…'
    actions.manager.state = 'idle'


def test_saved_budget_and_package_boundary_are_passed_to_prompt(qtbot, opened):
    controller, gui, actions = opened
    gui.state['SummaryStore'] = {'max_bytes': 128 * 1073741824}
    actions.build_dialog()
    qtbot.waitUntil(lambda: actions.dialog is not None)
    dialog = actions.dialog
    assert dialog.budget.value() == 128
    dialog.destination.setText(str(controller.dir_path / 'inside-store'))
    qtbot.waitUntil(lambda: 'outside the bounded package' in dialog.problem.text())
    assert not dialog.build_button.isEnabled()


def test_automatic_prompt_occurs_once_per_open(qtbot, opened):
    _, _, actions = opened
    actions._checked = False
    actions._automatic_check()
    qtbot.waitUntil(lambda: actions.dialog is not None)
    first = actions.dialog
    first._continue()
    actions._automatic_check()
    assert actions.dialog is first
    assert not first.isVisible()


def test_destination_preflight_does_not_block_gui_or_apply_stale_path(
    qtbot, tmp_path, monkeypatch
):
    started, release = Event(), Event()
    original = summary_task_gui.check_destination

    def blocked(request, context):
        if not started.is_set():
            started.set()
            assert release.wait(5)
        return original(request, context)

    monkeypatch.setattr(summary_task_gui, 'check_destination', blocked)
    estimate = StoreEstimate(100, 100, 10, 6, str(tmp_path / 'source'))
    prompt = summary_task_gui.StorePrompt(
        StoreInspection(estimate, False, True), tmp_path / 'new-store'
    )
    heartbeat = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: heartbeat.append(True))
    timer.start()
    try:
        qtbot.waitUntil(started.is_set)
        prompt.destination.setText(str(tmp_path))
        qtbot.waitUntil(lambda: len(heartbeat) >= 5)
        release.set()
        qtbot.waitUntil(lambda: 'new directory' in prompt.problem.text())
        assert not prompt.build_button.isEnabled()
    finally:
        release.set()
        timer.stop()
        prompt.close()


def test_vetoed_close_keeps_task_alive(qtbot, opened, tmp_path, monkeypatch):
    from phylib.utils import connect, unconnect

    _, gui, actions = opened
    started, release = Event(), Event()

    def blocked(request, context):
        started.set()
        assert release.wait(5)
        context.check_cancel()

    monkeypatch.setattr(summary_task_gui, 'prepare_summary_store', blocked)

    @connect(event='close', sender=gui)
    def veto(sender):
        return False

    try:
        actions.build(tmp_path / 'store')
        qtbot.waitUntil(started.is_set)
        gui.close()
        assert not gui._closed
        assert not actions._closed
        assert actions.manager.busy
        assert not actions.manager._cancel.is_set()
    finally:
        unconnect(veto)
        actions.manager.cancel()
        release.set()
        qtbot.waitUntil(lambda: not actions.manager.busy)


def test_gui_deletion_detaches_worker_before_controller_cleanup(qtbot, tmp_path, monkeypatch):
    from PyQt5 import sip
    from PyQt5.QtCore import QCoreApplication, QEvent

    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    actions = controller._summary_store_actions
    actions._checked = True
    started, release = Event(), Event()

    def blocked(request, context):
        started.set()
        assert release.wait(5)
        context.check_cancel()

    monkeypatch.setattr(summary_task_gui, 'prepare_summary_store', blocked)
    try:
        actions.build(tmp_path / 'store')
        qtbot.waitUntil(started.is_set)
        gui.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(gui)
        assert actions._closed
        assert actions.manager._cancel.is_set()
        release.set()
        qtbot.waitUntil(lambda: not actions.manager.busy)
    finally:
        release.set()
        if not sip.isdeleted(gui):
            gui.close()
        controller.close()


def test_automatic_prompt_keeps_main_window_shortcut_ownership(qtbot, opened):
    from phy.gui.qt import QApplication

    _, gui, actions = opened
    actions._checked = False
    gui.show()
    qtbot.waitUntil(lambda: actions.dialog is not None and actions.dialog.isVisible())
    qtbot.waitUntil(lambda: QApplication.activeWindow() is gui)
    assert actions.dialog.isVisible()


def test_prompt_disables_build_with_insufficient_free_space(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(summary_task_gui, '_free_space', lambda path: 0)
    estimate = StoreEstimate(100, 100, 10, 6, str(tmp_path / 'source'))
    prompt = summary_task_gui.StorePrompt(
        StoreInspection(estimate, False, True), tmp_path / 'store'
    )
    try:
        qtbot.waitUntil(lambda: 'not enough free space' in prompt.problem.text())
        assert not prompt.build_button.isEnabled()
        assert '0.00 GB' in prompt.available.text()
    finally:
        prompt.close()


def test_existing_row_store_upgrades_without_source_or_row_rebuild(
    qtbot, opened, tmp_path, monkeypatch
):
    from phy.apps.large_recording.live_summaries import LiveSummaryEngine
    from phy.apps.large_recording.range_index import AmplitudeRangeIndex
    from phy.apps.large_recording.summary_store import build_summary_store

    controller, gui, actions = opened
    store = build_summary_store(controller.dir_path, tmp_path / 'store')
    source = controller.bounded_session.manifest['source']['path']
    from pathlib import Path

    Path(source).rename(tmp_path / 'offline-source')

    def forbid(*args, **kwargs):
        raise AssertionError('Upgrade rebuilt existing rows')

    monkeypatch.setattr(summary_task_gui, 'prepare_summary_store', forbid)
    actions.build_dialog()
    qtbot.waitUntil(lambda: actions.dialog is not None)
    assert actions._inspection.compatible
    assert not actions._inspection.optimized
    assert actions._inspection.source_available
    actions.dialog.close()
    actions.build(store / 'amplitude-index')
    qtbot.waitUntil(lambda: actions._mode == 'attach-range' and not actions.manager.busy)
    with LiveSummaryEngine(controller.dir_path) as engine:
        index = AmplitudeRangeIndex(engine)
        index.close()
    assert not controller._live_summaries._suspended


def test_cancelling_amplitude_preparation_preserves_usable_rows(
    qtbot, opened, tmp_path, monkeypatch
):
    controller, gui, actions = opened
    started, release = Event(), Event()

    def blocked(request, context):
        started.set()
        assert release.wait(5)
        context.check_cancel()

    monkeypatch.setattr(summary_task_gui, 'prepare_amplitude_index', blocked)
    controller._live_summaries._errors['stale-error'] = 'unavailable: old source error'
    try:
        actions.build(tmp_path / 'store')
        qtbot.waitUntil(started.is_set)
        assert summary_store_reference(controller.dir_path) is not None
        actions.manager.cancel()
        release.set()
        qtbot.waitUntil(lambda: not actions.manager.busy)
        assert actions.manager.state == 'cancelled'
        assert not controller._live_summaries._errors
        assert not controller._live_summaries._suspended
        children = controller.supervisor.split_source_amplitude(0, 4).added
        qtbot.waitUntil(lambda: controller.get_full_count(children[0]) == 2)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not actions.manager.busy)


def test_late_row_commit_cancel_stops_next_long_stage(qtbot, opened, tmp_path, monkeypatch):
    controller, gui, actions = opened
    original = summary_task_gui.prepare_summary_store
    committed, release = Event(), Event()
    calls = []

    def controlled(request, context):
        result = original(request, context)
        committed.set()
        assert release.wait(5)
        return result

    def forbid(*args):
        calls.append(True)
        raise AssertionError('Late cancel started another long stage')

    monkeypatch.setattr(summary_task_gui, 'prepare_summary_store', controlled)
    monkeypatch.setattr(summary_task_gui, 'prepare_amplitude_index', forbid)
    try:
        actions.build(tmp_path / 'store')
        qtbot.waitUntil(committed.is_set)
        actions.manager.cancel()
        release.set()
        qtbot.waitUntil(lambda: actions._mode == 'attach' and not actions.manager.busy)
        assert not calls
        assert summary_store_reference(controller.dir_path) is not None
        assert not controller._live_summaries._suspended
    finally:
        release.set()
        qtbot.waitUntil(lambda: not actions.manager.busy)


def test_attach_independent_amplitude_index_from_gui(qtbot, opened, tmp_path):
    from phy.apps.large_recording.live_summaries import LiveSummaryEngine
    from phy.apps.large_recording.range_build import build_amplitude_index
    from phy.apps.large_recording.range_index import AmplitudeRangeIndex
    from phy.apps.large_recording.summary_store import attach_summary_store, build_summary_store

    controller, gui, actions = opened
    store = build_summary_store(controller.dir_path, tmp_path / 'store')
    index = build_amplitude_index(controller.dir_path, tmp_path / 'independent-index')
    attach_summary_store(controller.dir_path, store)
    moved = tmp_path / 'moved-index'
    index.rename(moved)
    actions.attach_index(moved)
    qtbot.waitUntil(lambda: not actions.manager.busy)
    assert actions.manager.state == 'succeeded'
    with LiveSummaryEngine(controller.dir_path) as engine:
        reader = AmplitudeRangeIndex(engine)
        assert reader.path == moved
        reader.close()
    assert not controller._live_summaries._suspended
