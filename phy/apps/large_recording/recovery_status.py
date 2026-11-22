"""A passive draft-protection indicator and user-controlled checkpoint reminder."""

import logging
import math
import time
from contextlib import suppress
from datetime import datetime

from phylib.utils import connect, unconnect

from phy.gui.qt import (
    QActionGroup,
    QApplication,
    QEvent,
    QHBoxLayout,
    QKeySequence,
    QLabel,
    QMenu,
    QPushButton,
    Qt,
    QTimer,
    QWidget,
    QWindow,
)

logger = logging.getLogger(__name__)
_NORMAL = 'QPushButton {background:#292d32;color:#eeeeee;border:1px solid #666;padding:3px 8px;}'
_REMINDER = (
    'QPushButton {background:#e6bd6b;color:#20180a;border:1px solid #e6bd6b;padding:3px 8px;}'
)
_INPUT_EVENTS = {
    QEvent.KeyPress,
    QEvent.MouseButtonPress,
    QEvent.MouseMove,
    QEvent.Wheel,
}


def _timestamp(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _age_label(saved_at, now):
    if saved_at is None:
        return 'No checkpoint yet'
    seconds = max(0, now - saved_at)
    if seconds < 60:
        return 'Checkpoint just now'
    if seconds < 3600:
        return f'Checkpoint {int(seconds // 60)} min ago'
    if seconds < 86400:
        return f'Checkpoint {int(seconds // 3600)} h ago'
    return f'Checkpoint {int(seconds // 86400)} days ago'


class CheckpointStatus(QWidget):
    """Keep reminders outside plots and never trigger a save from a timer.

    The recovery coordinator owns the durability acknowledgements. This widget
    only reads its small status dictionary; it never captures or serializes data.
    """

    def __init__(self, controller, gui, *, clock=time.monotonic, wall_clock=time.time):
        super().__init__(gui)
        self.controller = controller
        self.gui = gui
        self.recovery = controller.recovery
        self._clock, self._wall_clock = clock, wall_clock
        self._closed = False
        self._dirty = False
        self._dirty_since = None
        self._saved_at = None
        self._observed_revision = None
        self._snoozed_until = 0
        self._last_input = self._clock()
        self._highlighted = False
        self._checkpoint_error = None
        preferences = gui.state.get('CheckpointReminder', {})
        self.reminders_enabled = bool(preferences.get('enabled', True))
        minutes = preferences.get('interval_minutes', 15)
        self.interval_minutes = minutes if minutes in (5, 15, 30) else 15

        self.label = QLabel(self)
        self.label.setObjectName('checkpointProtectionStatus')
        shortcut = QKeySequence('Ctrl+S').toString(QKeySequence.NativeText)
        for actions in gui.actions:
            save = actions.get('save')
            if save is not None and not save.shortcut().isEmpty():
                shortcut = save.shortcut().toString(QKeySequence.NativeText)
                break
        self.save_button = QPushButton(f'Save checkpoint · {shortcut}', self)
        self.save_button.setObjectName('saveCheckpointButton')
        self.save_button.setFocusPolicy(Qt.NoFocus)
        self.save_button.setStyleSheet(_NORMAL)
        self.save_button.clicked.connect(self._save)
        self.save_button.setAccessibleName('Save checkpoint')
        self.options = QPushButton(self)
        self.options.setObjectName('checkpointReminderOptions')
        self.options.setText('⋮')
        self.options.setAccessibleName('Checkpoint reminder options')
        self.options.setToolTip('Checkpoint reminder options')
        self.options.setFocusPolicy(Qt.NoFocus)
        self.options.setStyleSheet('QPushButton::menu-indicator {image:none; width:0px;}')
        menu = QMenu(self.options)
        self.reminder_action = menu.addAction('Checkpoint reminders')
        self.reminder_action.setCheckable(True)
        self.reminder_action.setChecked(self.reminders_enabled)
        self.reminder_action.toggled.connect(self.set_reminders_enabled)
        self.snooze_action = menu.addAction('Remind me later')
        self.snooze_action.triggered.connect(self.snooze)
        intervals = menu.addMenu('Remind after')
        self.interval_group = QActionGroup(self)
        self.interval_group.setExclusive(True)
        self.interval_actions = {}
        for minutes in (5, 15, 30):
            action = intervals.addAction(f'{minutes} minutes of unsaved curation')
            action.setCheckable(True)
            action.setChecked(minutes == self.interval_minutes)
            self.interval_group.addAction(action)
            action.triggered.connect(lambda checked, value=minutes: self.set_interval(value))
            self.interval_actions[minutes] = action
        menu.addSeparator()
        menu.addAction('Retry draft protection').triggered.connect(self.recovery.request_flush)
        self.options.setMenu(menu)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 2, 0)
        layout.setSpacing(8)
        layout.addWidget(self.label)
        layout.addWidget(self.save_button)
        layout.addWidget(self.options)
        # Reserve the same area before and after highlighting or changing status.
        metrics = self.label.fontMetrics()
        self.label.setFixedWidth(
            max(
                metrics.horizontalAdvance(f'{state} · Checkpoint 999 min ago')
                for state in (
                    'Draft protected ✓',
                    'Protecting draft…',
                    'Protection failed',
                    'Draft protection ready',
                    'Draft protection off',
                    'Checkpoint failed',
                )
            )
            + 8
        )
        self.save_button.setFixedWidth(self.save_button.sizeHint().width() + 8)
        self.options.setFixedWidth(24)
        self.setFixedWidth(self.sizeHint().width())
        gui._status_bar.addPermanentWidget(self)
        self.recovery.changed.connect(self.refresh)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        QApplication.instance().installEventFilter(self)

        @connect(event='closed', sender=gui)
        def on_closed(sender):
            self.dispose()

        self._on_closed = on_closed
        gui.destroyed.connect(self.dispose)
        self.refresh()

    def eventFilter(self, receiver, event):
        if not self._closed and event.type() in _INPUT_EVENTS:
            inside = isinstance(receiver, QWidget) and (
                receiver is self.gui or self.gui.isAncestorOf(receiver)
            )
            if isinstance(receiver, QWindow):
                window = receiver
                while window is not None:
                    if window is self.gui.windowHandle():
                        inside = True
                        break
                    window = window.parent()
            if inside:
                self._last_input = self._clock()
        return False

    def _tick(self):
        self.refresh(read_dirty=False)

    def refresh(self, *, read_dirty=True):
        if self._closed:
            return
        status = self.recovery.status
        previous = self._highlighted
        now = self._clock()
        revision = status.get('current_revision', 0)
        saved_at = _timestamp(status.get('last_checkpoint_at'))
        saved_changed = saved_at != self._saved_at
        if read_dirty or revision != self._observed_revision or saved_changed:
            dirty = bool(self.controller.supervisor.is_dirty())
            if dirty and (self._dirty_since is None or saved_changed):
                self._dirty_since = now
            elif not dirty:
                self._dirty_since = None
            self._dirty = dirty
        if saved_changed:
            self._checkpoint_error = None
            self._highlighted = False
        self._observed_revision = revision
        self._saved_at = saved_at
        enabled = status.get('enabled', True)
        state = status.get('state')
        if not enabled or state in ('disabled', 'closed'):
            protection = 'Draft protection off'
        elif state == 'error' or status.get('error'):
            protection = 'Protection failed'
        elif state == 'protected' and revision == status.get('protected_revision'):
            protection = 'Draft protected ✓'
        elif state in ('pending', 'saving') or revision != status.get('protected_revision'):
            protection = 'Protecting draft…'
        else:
            protection = 'Draft protection ready'
        if self._checkpoint_error:
            protection = 'Checkpoint failed'
        self.label.setText(f'{protection} · {_age_label(saved_at, self._wall_clock())}')
        tooltip = (
            'Draft protection preserves a recoverable working copy. Save checkpoint '
            'commits validated curation for reopening and apply-curation. '
            'Checkpoint reminders do not affect draft protection.'
        )
        protected_at = _timestamp(status.get('last_protected_at'))
        if protected_at is not None:
            try:
                stamp = (
                    datetime.fromtimestamp(protected_at).astimezone().isoformat(timespec='seconds')
                )
            except (OverflowError, OSError, ValueError):
                stamp = str(protected_at)
            tooltip += f' Last confirmed draft protection: {stamp}.'
        error = self._checkpoint_error or status.get('error')
        if error:
            tooltip += f' {str(error)[:2048]}'
        self.label.setToolTip(tooltip)
        self.save_button.setToolTip(tooltip + ' Save when you reach a useful stopping point.')
        eligible = (
            self.reminders_enabled
            and self._dirty
            and self._dirty_since is not None
            and now - self._dirty_since >= self.interval_minutes * 60
            and now >= self._snoozed_until
        )
        if not eligible:
            self._highlighted = False
        elif now - self._last_input >= 1:
            self._highlighted = True
        if previous != self._highlighted:
            self.save_button.setStyleSheet(_REMINDER if self._highlighted else _NORMAL)
        self.save_button.setProperty('checkpointReminder', self._highlighted)
        self.save_button.setAccessibleDescription(
            'Checkpoint reminder: save at a useful stopping point.' if self._highlighted else ''
        )
        self.save_button.setEnabled(state != 'closed' and (self._dirty or saved_at is None))

    def _save(self):
        if self._closed:
            return
        try:
            self.controller.supervisor.save()
        except Exception as error:
            self._checkpoint_error = str(error)
            logger.warning('Checkpoint save failed: %s', error)
        self.refresh()

    def _preferences(self):
        self.gui.state['CheckpointReminder'] = {
            'enabled': self.reminders_enabled,
            'interval_minutes': self.interval_minutes,
        }

    def set_reminders_enabled(self, enabled):
        self.reminders_enabled = bool(enabled)
        self.reminder_action.setChecked(self.reminders_enabled)
        self._preferences()
        self.refresh()

    def set_interval(self, minutes):
        if minutes not in (5, 15, 30):
            raise ValueError('Choose a 5, 15 or 30 minute checkpoint reminder interval.')
        self.interval_minutes = minutes
        self.interval_actions[minutes].setChecked(True)
        self._preferences()
        self.refresh()

    def snooze(self):
        self._snoozed_until = self._clock() + self.interval_minutes * 60
        self.refresh()

    def dispose(self, *args):
        if self._closed:
            return
        self._closed = True
        unconnect(self._on_closed)
        for cleanup in (
            self._timer.stop,
            lambda: self.recovery.changed.disconnect(self.refresh),
            lambda: QApplication.instance().removeEventFilter(self),
        ):
            with suppress(RuntimeError, TypeError):
                cleanup()


def attach_checkpoint_status(controller, gui):
    """Attach one status area to a bounded GUI after recovery is initialized."""
    widget = getattr(gui, '_checkpoint_status', None)
    if widget is None:
        widget = gui._checkpoint_status = CheckpointStatus(controller, gui)
    return widget
