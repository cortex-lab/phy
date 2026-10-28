"""Dataset-specific store controls on top of reusable background task UI."""

import shutil
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from phylib.utils import connect, unconnect
from PyQt5 import sip
from PyQt5.QtWidgets import QFileDialog

from phy.gui.qt import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QObject,
    QPushButton,
    QSpinBox,
    Qt,
    QTimer,
    QVBoxLayout,
    pyqtSignal,
)
from phy.gui.task_widgets import TaskWidget
from phy.gui.tasks import BackgroundTaskManager, TaskSpec

from .summary_store import summary_store_reference
from .summary_tasks import (
    StoreRequest,
    connect_summary_store,
    inspect_summary_store,
    prepare_summary_store,
)


def _free_space(path):
    path = Path(path).expanduser().resolve().parent
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


@dataclass(frozen=True)
class DestinationRequest:
    text: str
    package: str = ''
    source: str = ''


@dataclass(frozen=True)
class DestinationStatus:
    text: str
    free_bytes: int
    exists: bool
    problem: str = ''


def check_destination(request, context):
    context.check_cancel()
    path = Path(request.text).expanduser()
    free = _free_space(path)
    exists = path.exists()
    context.check_cancel()
    problem = ''
    destination = path.resolve()
    for label, root in [('bounded package', request.package), ('original source', request.source)]:
        if root:
            root = Path(root).resolve()
            if destination == root or root in destination.parents:
                problem = f'Choose a destination outside the {label}.'
    return DestinationStatus(request.text, free, exists, problem)


class StorePrompt(QDialog):
    """Nonmodal opt-in; no source scans or allocation occur in this dialog."""

    build_requested = pyqtSignal(str, bool, object)
    continued = pyqtSignal(bool)

    def __init__(
        self,
        inspection,
        destination,
        suppressed=False,
        parent=None,
        max_bytes=68719476736,
        package='',
    ):
        super().__init__(parent)
        self.inspection = inspection
        self.package = package
        self._preflight_closed = False
        self._disk_key = uuid4().hex
        self._disk_manager = BackgroundTaskManager()
        self._disk_manager.finished.connect(self._disk_finished)
        self._disk_timer = QTimer(self)
        self._disk_timer.setSingleShot(True)
        self._disk_timer.setInterval(250)
        self._disk_timer.timeout.connect(self._disk_check)
        self.finished.connect(self._dispose_preflight)
        self.setWindowTitle('Prepare faster full-recording summaries?')
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        layout = QVBoxLayout(self)
        text = 'Build a reusable store so splits read only the affected original units. You can keep curating while it runs.'
        layout.addWidget(QLabel(text, self))
        layout.addWidget(
            QLabel(
                f'Estimated additional space: {inspection.estimate.estimated_bytes / 1e9:.2f} GB',
                self,
            )
        )
        row = QHBoxLayout()
        self.destination = QLineEdit(str(destination), self)
        self.choose = QPushButton('Choose parent folder…', self)
        row.addWidget(self.destination)
        row.addWidget(self.choose)
        layout.addLayout(row)
        self.budget = QSpinBox(self)
        self.budget.setRange(1, 1048576)
        self.budget.setValue(max(1, int(max_bytes) // 1073741824))
        self.budget.setSuffix(' GiB store budget')
        self.budget.valueChanged.connect(self._check)
        layout.addWidget(self.budget)
        self.available = QLabel('', self)
        self.problem = QLabel('', self)
        layout.addWidget(self.available)
        layout.addWidget(self.problem)
        self.suppress = QCheckBox('Don’t ask again for this dataset', self)
        self.suppress.setChecked(suppressed)
        layout.addWidget(self.suppress)
        buttons = QHBoxLayout()
        self.build_button = QPushButton('Build in background', self)
        self.continue_button = QPushButton('Continue without', self)
        buttons.addWidget(self.build_button)
        buttons.addWidget(self.continue_button)
        layout.addLayout(buttons)
        self.choose.clicked.connect(self._choose)
        self.destination.textChanged.connect(self._check)
        self.build_button.clicked.connect(self._build)
        self.continue_button.clicked.connect(self._continue)
        self._check()
        self.resize(650, 220)

    def _choose(self):
        parent = QFileDialog.getExistingDirectory(
            self, 'Choose parent folder for the new summary store'
        )
        if parent:
            self.destination.setText(str(Path(parent) / 'summary-store'))

    def _check(self):
        if self._preflight_closed:
            return
        self.build_button.setEnabled(False)
        self.available.setText('Checking available space…')
        self.problem.setText('')
        if self._disk_manager.busy:
            self._disk_manager.cancel()
        self._disk_timer.start()

    def _disk_check(self):
        if self._disk_manager.busy:
            return  # Its terminal callback starts only the latest requested path.
        text = self.destination.text()
        if not text.strip():
            self.problem.setText('Choose a destination for the new store.')
            return
        self._disk_manager.submit(
            TaskSpec(
                f'destination:{self._disk_key}',
                'Checking destination',
                check_destination,
                DestinationRequest(text, self.package, self.inspection.estimate.source_path),
                heavy=False,
            )
        )

    def _disk_finished(self, outcome):
        if self._preflight_closed:
            return
        if self._disk_manager.spec.request.text != self.destination.text():
            self._disk_timer.start(0)
            return
        if outcome.state == 'cancelled':
            self._disk_timer.start(0)
            return
        if outcome.state == 'failed':
            self.problem.setText(outcome.error)
            return
        result = outcome.result
        self.available.setText(f'Available space: {result.free_bytes / 1e9:.2f} GB')
        problem = result.problem
        if result.exists:
            problem = (
                'Choose a new directory. Use File > Attach summary store for an existing store.'
            )
        elif result.free_bytes < self.inspection.estimate.estimated_bytes:
            problem = 'There is not enough free space at this location.'
        elif self.inspection.estimate.estimated_bytes > self.budget.value() * 1073741824:
            problem = 'The estimate exceeds the store budget. Increase the budget to build.'
        elif not self.inspection.source_available:
            problem = 'Source unavailable. Use File > Locate summary source, or attach an existing store.'
        self.problem.setText(problem)
        self.build_button.setEnabled(not problem)

    def _dispose_preflight(self, *args):
        if self._preflight_closed:
            return
        self._preflight_closed = True
        self._disk_timer.stop()
        self._disk_manager.finished.disconnect(self._disk_finished)
        self._disk_manager.detach()

    def _build(self):
        if self.build_button.isEnabled():
            self.build_requested.emit(
                self.destination.text(),
                self.suppress.isChecked(),
                self.budget.value() * 1073741824,
            )
            self.accept()

    def _continue(self):
        self.continued.emit(self.suppress.isChecked())
        self.reject()


class SummaryStoreActions(QObject):
    """Own one GUI generation; workers receive only frozen request snapshots."""

    def __init__(self, controller, gui):
        super().__init__(gui)
        self.controller, self.gui = controller, gui
        self.package = str(controller.dir_path)
        self.service = controller._live_summaries
        self.manager = BackgroundTaskManager()
        self.widget = TaskWidget(self.manager, gui)
        gui._status_bar.addPermanentWidget(self.widget, 1)
        self.widget.hide()
        self.dialog = None
        self._closed = self._checked = self._paused = False
        self._manual = False
        self._mode = None
        self._request = None
        self._binding = controller.bounded_session.manifest.get('preparation_identity', '')
        gui.state.add_local_keys(
            ['SummaryStore.suppress_prompt', 'SummaryStore.destination', 'SummaryStore.max_bytes']
        )
        self.manager.finished.connect(self._finished)
        # Some embedders delete the GUI before closing its controller.
        gui.destroyed.connect(lambda: self.close())

        @gui.file_actions.add
        def build_summary_store():
            """Prepare a reusable summary store in the background."""
            self.build_dialog()

        @gui.file_actions.add
        def attach_summary_store():
            """Attach an existing compatible store, including a relocated one."""
            if self.manager.busy:
                self.widget.show()
                return
            path = QFileDialog.getExistingDirectory(gui, 'Attach summary store')
            if path:
                self.attach(path)

        @gui.file_actions.add
        def show_background_task():
            """Show progress and details for the current background task."""
            self.widget.show()

        @connect(event='closed', sender=gui)
        def on_closed(sender):
            self.close()

        self._on_closed = on_closed

        @connect(event='show', sender=gui)
        def on_shown(sender):
            QTimer.singleShot(0, self._automatic_check)

        self._on_shown = on_shown
        if gui.isVisible():
            QTimer.singleShot(0, self._automatic_check)

    def _valid(self):
        return (
            not self._closed
            and not self.gui._closed
            and self.controller.bounded_session.manifest.get('preparation_identity', '')
            == self._binding
        )

    def _automatic_check(self):
        if self._checked or not self._valid():
            return
        self._checked = True
        if not self.gui.state.get('SummaryStore', {}).get('suppress_prompt', False):
            self._inspect()

    def _request_for(self, destination=''):
        try:
            source = self.gui.state.get('LiveSummaries', {}).get('source_path')
            return StoreRequest(
                self.package,
                source,
                destination,
                summary_store_reference(self.package, allow_invalid=True),
                dataset_id=self._binding,
            )
        except (OSError, ValueError) as error:
            self.widget.notice(f'Could not read summary-store settings: {error}')
            return None

    def _submit(self, mode, worker, request, heavy):
        if request is None:
            self._resume()
            return
        self._mode, self._request = mode, request
        try:
            self.manager.submit(
                TaskSpec(
                    f'summary-store:{mode}:{self.package}:{self._binding}',
                    {
                        'inspect': 'Checking summary store',
                        'build': 'Preparing faster summaries',
                        'attach': 'Attaching summary store',
                    }[mode],
                    worker,
                    request,
                    heavy=heavy,
                )
            )
        except (ValueError, TypeError, OSError) as error:
            self.widget.notice(str(error))
            self._resume()

    def _inspect(self):
        if not self._valid() or self.manager.busy:
            return
        self._submit('inspect', inspect_summary_store, self._request_for(), False)

    def build_dialog(self):
        if not self._valid():
            return
        if self.dialog is not None and self.dialog.isVisible():
            self.dialog.raise_()
            self.dialog.activateWindow()
            return
        self._manual = True
        if self.manager.busy:
            if self._mode != 'inspect':
                self.widget.show()
            return
        self._inspect()

    def _remember(self, suppress, destination=None):
        state = dict(self.gui.state.get('SummaryStore', {}))
        state['suppress_prompt'] = bool(suppress)
        if destination is not None:
            state['destination'] = str(destination)
        self.gui.state['SummaryStore'] = state

    def build(self, destination, suppress=False, max_bytes=68719476736):
        if not self._valid() or self.manager.busy:
            return
        self._remember(suppress, destination)
        self.gui.state['SummaryStore']['max_bytes'] = int(max_bytes)
        request = self._request_for(str(Path(destination).expanduser().absolute()))
        if request is None:
            return
        request = replace(request, max_bytes=int(max_bytes))
        self._paused = True
        self.service.pause('waiting for summary-store preparation')
        self._submit('build', prepare_summary_store, request, True)

    def attach(self, path):
        if not self._valid() or self.manager.busy:
            return
        self._paused = True
        self.service.pause('waiting for summary-store attachment')
        self._submit(
            'attach',
            connect_summary_store,
            self._request_for(str(Path(path).expanduser().absolute())),
            False,
        )

    def _resume(self):
        if self._paused and self._valid():
            self._paused = False
            self.service.resume()

    def _finished(self, outcome):
        if not self._valid():
            return
        mode, request = self._mode, self._request
        if outcome.state != 'succeeded':
            self._resume()
            if mode == 'attach':
                self.widget.notice(
                    f'Store remains at {request.destination}; attachment did not complete. See Details.'
                )
            return
        if mode == 'inspect':
            self.widget.hide()
            inspection = outcome.result
            if inspection.compatible and not self._manual:
                return
            state = self.gui.state.get('SummaryStore', {})
            destination = state.get('destination', self.package + '.summary-store')
            self.dialog = StorePrompt(
                inspection,
                destination,
                state.get('suppress_prompt', False),
                self.gui,
                max_bytes=state.get('max_bytes', 68719476736),
                package=self.package,
            )
            self.dialog.build_requested.connect(self.build)
            self.dialog.continued.connect(self._remember)
            active, focus = QApplication.activeWindow(), QApplication.focusWidget()
            self.dialog.setAttribute(Qt.WA_ShowWithoutActivating, not self._manual)
            self.dialog.show()
            if self._manual:
                self.dialog.raise_()
                self.dialog.activateWindow()
            else:
                # Native platform activation may arrive after QWidget.show().
                QTimer.singleShot(0, lambda: self._restore_activation(active, focus))
            self._manual = False
        elif mode == 'build':
            artifact = outcome.result
            if artifact.warning:
                self.widget.notice(
                    f'Store created at {artifact.path}; finalization needs attention. See Details.'
                )
                self.widget.log.appendPlainText(artifact.warning)
                self._resume()
            else:
                self._submit(
                    'attach',
                    connect_summary_store,
                    replace(request, destination=artifact.path),
                    False,
                )
        elif mode == 'attach':
            self.service.retry()
            self._resume()
            self.widget.notice('Summary store ready. New splits use the prepared store.')

    def _restore_activation(self, active, focus):
        if not self._valid() or QApplication.activeWindow() not in (self.dialog, None):
            return
        target = active if active is not None and not sip.isdeleted(active) else self.gui
        target.activateWindow()
        if target.windowHandle() is not None:
            target.windowHandle().requestActivate()
        QApplication.setActiveWindow(target)
        if focus is not None and not sip.isdeleted(focus):
            focus.setFocus(Qt.OtherFocusReason)

    def close(self):
        if self._closed:
            return
        self._closed = True
        with suppress(TypeError, RuntimeError):
            self.manager.finished.disconnect(self._finished)
        self.widget.dispose()
        self.manager.detach()
        if self.dialog is not None and not sip.isdeleted(self.dialog):
            self.dialog.close()
        unconnect(self._on_closed, self._on_shown)
