"""Checkpoint reminders never publish data, interrupt input, or invent protection."""

from types import SimpleNamespace

import pytest
from phylib.utils import connect, unconnect

from phy.apps.large_recording.recovery_status import CheckpointStatus
from phy.gui import GUI
from phy.gui.qt import QApplication, QLineEdit, QObject, Qt, pyqtSignal


class Clock:
    value = 0

    def monotonic(self):
        return self.value

    def time(self):
        return 1_700_000_000 + self.value

    def advance(self, seconds):
        self.value += seconds


class Recovery(QObject):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.value = {
            'enabled': True,
            'state': 'protected',
            'current_revision': 0,
            'protected_revision': 0,
            'last_protected_at': None,
            'last_checkpoint_at': 1_700_000_000,
            'error': None,
        }
        self.flushes = 0

    @property
    def status(self):
        return dict(self.value)

    def update(self, **values):
        self.value.update(values)
        self.changed.emit()

    def request_flush(self):
        self.flushes += 1


class Supervisor:
    dirty = False
    saves = 0
    reads = 0
    save_error = None

    def is_dirty(self):
        self.reads += 1
        return self.dirty

    def save(self):
        self.saves += 1
        if self.save_error:
            raise self.save_error


@pytest.fixture
def status(qtbot, tmp_path):
    gui = GUI(config_dir=tmp_path / 'config', size=(1100, 650))
    gui.set_default_actions()
    qtbot.addWidget(gui)
    controller = SimpleNamespace(recovery=Recovery(), supervisor=Supervisor())
    clock = Clock()
    widget = CheckpointStatus(controller, gui, clock=clock.monotonic, wall_clock=clock.time)
    widget._timer.stop()  # Test the real timer callback using a deterministic clock.
    with qtbot.waitExposed(gui):
        gui.show()
    gui.activateWindow()
    qtbot.waitUntil(gui.isActiveWindow)
    yield widget, controller, clock, gui
    widget.dispose()
    gui.close()


def _edit(controller, clock, **kwargs):
    controller.supervisor.dirty = True
    controller.recovery.update(
        current_revision=1,
        protected_revision=1,
        last_protected_at=clock.time(),
        **kwargs,
    )


def test_reminder_tracks_curation_and_waits_for_idle_without_stealing_focus(status, qtbot):
    widget, controller, clock, gui = status
    field = QLineEdit()
    gui.add_view(field)
    field.setFocus()
    qtbot.waitUntil(field.hasFocus)
    size = widget.size()
    initial_reads = controller.supervisor.reads
    clock.advance(3600)
    for _ in range(3):
        widget._tick()
    assert not widget.save_button.property('checkpointReminder')
    assert controller.supervisor.reads == initial_reads
    assert not controller.supervisor.saves
    _edit(controller, clock)
    clock.advance(899)
    widget._tick()
    assert not widget.save_button.property('checkpointReminder')
    clock.advance(2)
    qtbot.keyClicks(field, 'curation')
    widget._tick()
    assert not widget.save_button.property('checkpointReminder')  # Wait for a pause.
    clock.advance(1.1)
    widget._tick()
    assert widget.save_button.property('checkpointReminder')
    assert '#e6bd6b' in widget.save_button.styleSheet()
    assert widget.size() == size
    assert QApplication.focusWidget() is field
    qtbot.keyClicks(field, 'dD')
    widget._tick()
    assert widget.save_button.property('checkpointReminder')  # Remains visible while typing.
    assert field.text() == 'curationdD'
    assert not controller.supervisor.saves


def test_snooze_disable_interval_and_successful_checkpoint_reset(status):
    widget, controller, clock, gui = status
    _edit(controller, clock)
    widget.set_interval(5)
    clock.advance(301)
    widget._tick()
    assert widget.save_button.property('checkpointReminder')
    widget.snooze()
    assert not widget.save_button.property('checkpointReminder')
    clock.advance(299)
    widget._tick()
    assert not widget.save_button.property('checkpointReminder')
    clock.advance(2)
    widget._tick()
    assert widget.save_button.property('checkpointReminder')
    widget.set_reminders_enabled(False)
    assert not widget.save_button.property('checkpointReminder')
    assert gui.state['CheckpointReminder'] == {'enabled': False, 'interval_minutes': 5}
    assert controller.recovery.status['enabled']  # Disabling reminders preserves recovery.
    widget.set_reminders_enabled(True)
    assert widget.save_button.property('checkpointReminder')
    controller.supervisor.dirty = False
    controller.recovery.update(last_checkpoint_at=clock.time())
    assert not widget.save_button.property('checkpointReminder')
    assert 'Checkpoint just now' in widget.label.text()
    assert '#e6bd6b' not in widget.save_button.styleSheet()
    assert not widget.save_button.isEnabled()
    clock.advance(10000)
    widget._tick()
    assert not widget.save_button.property('checkpointReminder')
    assert not controller.supervisor.saves


def test_only_acknowledged_recovery_is_called_protected_and_save_errors_are_visible(status):
    widget, controller, clock, _ = status
    controller.supervisor.dirty = True
    controller.recovery.update(current_revision=23, protected_revision=17)
    assert 'Protecting draft' in widget.label.text()
    controller.recovery.update(state='error', error='Disk full', last_protected_at=clock.time())
    assert 'Protection failed' in widget.label.text()
    assert 'Disk full' in widget.label.toolTip()
    assert 'Last confirmed' in widget.label.toolTip()
    controller.supervisor.save_error = OSError('Read-only checkpoint destination')
    widget.save_button.click()
    assert controller.supervisor.saves == 1
    assert 'Checkpoint failed' in widget.label.text()
    assert 'Read-only checkpoint' in widget.label.toolTip()
    controller.supervisor.save_error = None
    widget.save_button.click()
    assert controller.supervisor.saves == 2
    assert 'Checkpoint failed' in widget.label.text()  # No invented success timestamp.
    clock.advance(1)
    controller.supervisor.dirty = False
    controller.recovery.update(
        state='protected', protected_revision=23, last_checkpoint_at=clock.time(), error=None
    )
    assert 'Draft protected' in widget.label.text()
    assert 'Checkpoint just now' in widget.label.text()


def test_no_checkpoint_is_explicit_and_preferences_restore(status):
    widget, controller, _, gui = status
    controller.recovery.update(last_checkpoint_at=None)
    assert 'No checkpoint yet' in widget.label.text()
    assert widget.save_button.isEnabled()
    widget.set_interval(30)
    widget.set_reminders_enabled(False)
    widget.dispose()
    restored = CheckpointStatus(controller, gui)
    try:
        restored._timer.stop()
        assert not restored.reminders_enabled
        assert restored.interval_minutes == 30
        assert not restored.reminder_action.isChecked()
        assert restored.interval_actions[30].isChecked()
    finally:
        restored.dispose()


def test_recovery_warnings_and_startup_failures_are_visible(status):
    widget, controller, _, _ = status
    controller.recovery.update(warning='Another session saved; reopen before checkpointing.')
    assert 'Draft needs attention' in widget.label.text()
    assert 'Another session saved' in widget.label.toolTip()
    controller.recovery.update(warning=None, startup_error='Recovery discovery failed: disk error')
    assert 'Protection failed' in widget.label.text()
    assert 'disk error' in widget.label.toolTip()


def test_undo_to_checkpoint_restarts_reminder_at_the_next_edit(status):
    widget, controller, clock, _ = status
    _edit(controller, clock)
    clock.advance(901)
    widget._tick()
    assert widget.save_button.property('checkpointReminder')
    # Undo may return to the saved state without changing the checkpoint stamp.
    controller.supervisor.dirty = False
    controller.recovery.update(current_revision=2)
    assert not widget.save_button.property('checkpointReminder')
    clock.advance(5000)
    controller.supervisor.dirty = True
    controller.recovery.update(current_revision=3)
    assert not widget.save_button.property('checkpointReminder')
    clock.advance(899)
    widget._tick()
    assert not widget.save_button.property('checkpointReminder')
    clock.advance(2)
    widget._tick()
    assert widget.save_button.property('checkpointReminder')


def test_cancelled_close_keeps_status_active_and_disposal_disconnects(status):
    widget, controller, clock, gui = status

    @connect(event='close', sender=gui)
    def cancel(sender):
        return False

    try:
        gui.close()
        assert not gui._closed
        assert not widget._closed
        _edit(controller, clock)
        assert 'Draft protected' in widget.label.text()
    finally:
        unconnect(cancel)
    widget.dispose()
    last = widget.label.text()
    controller.recovery.update(state='error', error='Late worker result')
    assert widget.label.text() == last
