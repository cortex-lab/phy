"""Reusable nonmodal task presentation; hiding never cancels work."""

from contextlib import suppress

from PyQt5 import sip
from PyQt5.QtWidgets import QProgressBar

from .qt import QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget
from .tasks import StageETA


def _remaining(seconds):
    if seconds < 60:
        return f'about {max(1, round(seconds))} s remaining in this stage'
    return f'about {max(1, round(seconds / 60))} min remaining in this stage'


class TaskWidget(QWidget):
    """Compact status-bar task display with bounded expandable details."""

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._disposed = False
        self.manager = manager
        self.eta = StageETA()
        self._last_stage = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        self.label = QLabel('Background task', self)
        self.label.setMaximumWidth(420)
        self.bar = QProgressBar(self)
        self.bar.setMaximumWidth(160)
        self.estimate = QLabel('', self)
        self.cancel_button = QPushButton('Cancel', self)
        self.details_button = QPushButton('Details', self)
        self.hide_button = QPushButton('Hide', self)
        for widget in (
            self.label,
            self.bar,
            self.estimate,
            self.cancel_button,
            self.details_button,
            self.hide_button,
        ):
            layout.addWidget(widget)
        self.details = QDialog(parent)
        self.details.setWindowTitle('Background task details')
        detail_layout = QVBoxLayout(self.details)
        self.log = QPlainTextEdit(self.details)
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(200)
        detail_layout.addWidget(self.log)
        self.details.resize(650, 250)
        self.cancel_button.clicked.connect(manager.cancel)
        self.details_button.clicked.connect(self.details.show)
        self.hide_button.clicked.connect(self.hide)
        manager.progress.connect(self.on_progress)
        manager.finished.connect(self.on_finished)
        manager.state_changed.connect(self.on_state)

    def on_state(self, state):
        if state == 'pending':
            self.eta.reset()
            self._last_stage = None
            self.label.setText(self.manager.spec.title)
            self.bar.setRange(0, 0)
            self.estimate.setText('Estimating time remaining…')
            self.show()
        self.cancel_button.setEnabled(state in ('pending', 'running'))
        if state == 'committing':
            self.estimate.setText('Finishing…')
        elif state == 'cancelling':
            self.estimate.setText('Cancelling…')

    def on_progress(self, progress):
        self.label.setText(progress.stage)
        if progress.stage != self._last_stage:
            self.log.appendPlainText(progress.stage)
            self._last_stage = progress.stage
        if progress.total and progress.done is not None:
            self.bar.setRange(0, 1000)
            self.bar.setValue(min(1000, int(1000 * progress.done / progress.total)))
            self.bar.setFormat('%p%')
        else:
            self.bar.setRange(0, 0)
        remaining = self.eta.update(progress)
        self.estimate.setText(
            _remaining(remaining)
            if remaining is not None
            else (progress.message or ('Estimating time remaining…' if progress.total else ''))
        )
        if self.manager.state == 'committing':
            self.estimate.setText('Finishing…')
        elif self.manager.state == 'cancelling':
            self.estimate.setText('Cancelling…')
        if progress.message:
            self.log.appendPlainText(progress.message)

    def on_finished(self, outcome):
        self.cancel_button.setEnabled(False)
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000 if outcome.state == 'succeeded' else 0)
        self.label.setText(
            {'succeeded': 'Complete', 'cancelled': 'Cancelled', 'failed': 'Task failed'}[
                outcome.state
            ]
        )
        self.estimate.setText('')
        if outcome.error:
            self.log.appendPlainText(outcome.error)
            self.estimate.setText('See Details')

    def notice(self, message):
        self.log.appendPlainText(str(message))
        self.label.setText(str(message)[:160])
        self.label.setToolTip(str(message))
        self.show()

    def dispose(self):
        if getattr(self, '_disposed', False):
            return
        self._disposed = True
        for name, slot in (
            ('progress', self.on_progress),
            ('finished', self.on_finished),
            ('state_changed', self.on_state),
        ):
            # Qt can disconnect receivers while deleting their parent GUI.
            with suppress(TypeError, RuntimeError):
                getattr(self.manager, name).disconnect(slot)
        if not sip.isdeleted(self.details):
            self.details.close()
