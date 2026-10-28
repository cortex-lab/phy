"""Small, bounded background tasks with cooperative cancellation and queued delivery.

Workers receive immutable requests and a TaskContext, never GUI/session objects.
No task queue, persistent jobs, or interruption of native operations is implied.
"""

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from uuid import uuid4

from .qt import QObject, Qt, QTimer, pyqtSignal, pyqtSlot

_HEAVY_IO = Lock()
_ACTIVE = set()
_ACTIVE_LOCK = Lock()


class TaskCancelledError(Exception):
    """Cancellation observed before entering a publication boundary."""


@contextmanager
def heavy_io(check_cancel):
    """Share one cancellable full-source I/O permit across task implementations."""
    while True:
        check_cancel()
        if _HEAVY_IO.acquire(timeout=0.05):
            break
    try:
        check_cancel()
        yield
    finally:
        _HEAVY_IO.release()


@dataclass(frozen=True)
class TaskProgress:
    stage: str
    done: int | None = None
    total: int | None = None
    unit: str = ''
    message: str = ''


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    state: str
    result: object = None
    error: str = ''


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    worker: object
    request: object
    heavy: bool = True


def _immutable(value):
    if value is None or isinstance(value, (str, bytes, int, float, bool, Path)):
        return
    if isinstance(value, (tuple, frozenset)):
        for item in value:
            _immutable(item)
        return
    if is_dataclass(value) and value.__dataclass_params__.frozen:
        for field in fields(value):
            _immutable(getattr(value, field.name))
        return
    raise TypeError('Task inputs and results must be immutable values, not GUI/session objects.')


class StageETA:
    """Smoothed per-stage estimates; clock injection makes timing deterministic."""

    def __init__(self, clock=monotonic):
        self.clock = clock
        self.reset()

    def reset(self):
        self.stage = None
        self.started = self.last_time = self.last_done = self.rate = None

    def update(self, progress):
        now = self.clock()
        stage = (progress.stage, progress.unit, progress.total)
        if stage != self.stage or (
            self.last_done is not None
            and progress.done is not None
            and progress.done < self.last_done
        ):
            self.reset()
            self.stage, self.started = stage, now
        if progress.done is None or progress.total is None or progress.total <= 0:
            return None
        if self.last_time is not None and now > self.last_time and progress.done > self.last_done:
            rate = (progress.done - self.last_done) / (now - self.last_time)
            self.rate = rate if self.rate is None else 0.25 * rate + 0.75 * self.rate
        self.last_time, self.last_done = now, progress.done
        if now - self.started < 2 or not self.rate or progress.done < 1:
            return None
        return max(0, (progress.total - progress.done) / self.rate)


class TaskContext:
    """Worker-side cancellation, latest-only progress, and publication state."""

    def __init__(self, cancelled, lock, mailbox):
        self.cancelled, self.lock, self.mailbox = cancelled, lock, mailbox
        self.committing = False

    def check_cancel(self):
        if not self.committing and self.cancelled.is_set():
            raise TaskCancelledError()

    def report(self, progress):
        if not isinstance(progress, TaskProgress):
            raise TypeError('Expected TaskProgress.')
        with self.lock:
            self.mailbox[0] = progress

    def commit(self):
        """Enter a short noncancellable publication boundary after a final check."""
        self.check_cancel()
        self.committing = True
        self.report(
            TaskProgress('Finalizing', message='Finishing; publication cannot be cancelled.')
        )


class _Delivery(QObject):
    terminal = pyqtSignal(object)


class BackgroundTaskManager(QObject):
    """One task per manager, no queued backlog, and process-wide key deduplication.

    Progress occupies one mailbox slot even when the GUI event loop is stalled.
    A worker retains manager/delivery ownership through terminal cleanup. Owners
    must disconnect their own callbacks on accepted closure, then call detach().
    """

    progress = pyqtSignal(object)
    finished = pyqtSignal(object)
    state_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.state = 'idle'
        self.task_id = None
        self.spec = None
        self._cancel = Event()
        self._lock = Lock()
        self._mailbox = [None]
        self._detached = False
        self._context = None
        self._delivery = _Delivery()
        self._delivery.terminal.connect(self._terminal, Qt.QueuedConnection)
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._poll)

    @property
    def busy(self):
        return self.state in ('pending', 'running', 'cancelling', 'committing')

    def _set_state(self, state):
        self.state = state
        if not self._detached:
            self.state_changed.emit(state)

    def submit(self, spec):
        if self.busy or self._detached:
            raise ValueError('A background task is already active or its owner has closed.')
        _immutable(spec.request)
        with _ACTIVE_LOCK:
            if spec.key in _ACTIVE:
                raise ValueError('This background task is already running.')
            _ACTIVE.add(spec.key)
        self.spec, self.task_id = spec, uuid4().hex
        self._cancel = Event()
        self._mailbox = [None]
        self._context = TaskContext(self._cancel, self._lock, self._mailbox)
        self._set_state('pending')
        self._timer.start()
        Thread(target=self._run, args=(spec, self.task_id, self._context), daemon=True).start()
        return self.task_id

    def _run(self, spec, task_id, context):
        try:
            if spec.heavy:
                context.report(
                    TaskProgress('Waiting', message='Waiting for other recording I/O to finish.')
                )
                with heavy_io(context.check_cancel):
                    result = spec.worker(spec.request, context)
            else:
                context.check_cancel()
                result = spec.worker(spec.request, context)
            _immutable(result)
            # A committed artifact remains success even if a late cancel arrives.
            context.check_cancel()
            outcome = TaskOutcome(task_id, 'succeeded', result=result)
        except TaskCancelledError:
            outcome = TaskOutcome(task_id, 'cancelled')
        except Exception as error:
            outcome = TaskOutcome(task_id, 'failed', error=str(error))
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.discard(spec.key)
        self._delivery.terminal.emit(outcome)

    @pyqtSlot()
    def _poll(self):
        with self._lock:
            progress, self._mailbox[0] = self._mailbox[0], None
        if progress is None or self._detached:
            return
        if self._context.committing:
            self._set_state('committing')
        elif self.state != 'cancelling':
            self._set_state('running')
        self.progress.emit(progress)

    @pyqtSlot(object)
    def _terminal(self, outcome):
        if outcome.task_id != self.task_id:
            return
        self._poll()
        self._timer.stop()
        self._set_state(outcome.state)
        if not self._detached:
            self.finished.emit(outcome)

    def cancel(self):
        if self.busy:
            self._cancel.set()
            self._set_state('committing' if self._context.committing else 'cancelling')

    def detach(self):
        self._detached = True
        self.cancel()
        self._timer.stop()
