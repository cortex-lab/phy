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

from .qt import QObject, QTimer, pyqtSignal, pyqtSlot

_HEAVY_IO = Lock()
_ACTIVE = set()
_MANAGERS = {}
_ACTIVE_LOCK = Lock()


class TaskCancelledError(Exception):
    """Cancellation observed before entering a publication boundary."""


@contextmanager
def heavy_io(check_cancel, waiting=None):
    """Share one cancellable full-source I/O permit across task implementations."""
    notified = False
    while True:
        check_cancel()
        if _HEAVY_IO.acquire(timeout=0.05):
            break
        if waiting is not None and not notified:
            waiting()
            notified = True
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
        if self.last_done is None or progress.done > self.last_done:
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
        _immutable(progress)
        if not all(isinstance(v, str) for v in (progress.stage, progress.unit, progress.message)):
            raise TypeError('Progress labels must be strings.')
        if progress.done is not None and (type(progress.done) is not int or progress.done < 0):
            raise ValueError('Progress done must be a nonnegative integer.')
        if progress.total is not None and (type(progress.total) is not int or progress.total < 0):
            raise ValueError('Progress total must be a nonnegative integer.')
        with self.lock:
            self.mailbox[0] = progress

    def commit(self):
        """Enter a short noncancellable publication boundary after a final check."""
        with self.lock:
            if self.cancelled.is_set():
                raise TaskCancelledError()
            self.committing = True
        self.report(
            TaskProgress('Finalizing', message='Finishing; publication cannot be cancelled.')
        )


@dataclass
class _TaskRecord:
    """Plain Python worker ownership, independent of the manager's Qt lifetime."""

    task_id: str
    context: TaskContext
    outcome: TaskOutcome | None = None


class _ManagerLifetime:
    """The destruction callback captures no QObject or bound Qt method."""

    def __init__(self):
        self.dead = False
        self.record = None

    def destroy(self, *args):
        self.dead = True
        record = self.record
        if record is not None:
            with record.context.lock:
                record.context.cancelled.set()
            # QObject.destroyed runs on its GUI owner thread. Workers retain
            # only the plain record, so releasing this wrapper here is safe.
            _MANAGERS.pop(record.task_id, None)


def _run_task(spec, record):
    """A worker never retains, calls, or releases a Qt wrapper."""
    context = record.context
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
        context.check_cancel()
        outcome = TaskOutcome(record.task_id, 'succeeded', result=result)
    except TaskCancelledError:
        outcome = TaskOutcome(record.task_id, 'cancelled')
    except BaseException as error:
        outcome = TaskOutcome(record.task_id, 'failed', error=str(error))
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.discard(spec.key)
    with context.lock:
        record.outcome = outcome


class BackgroundTaskManager(QObject):
    """One task per manager, no backlog, and process-wide key deduplication.

    Workers retain only plain Python state. The GUI polls bounded progress and
    terminal mailboxes; it owns Qt cleanup even after the task owner detaches.
    QObject destruction cancels work and releases manager ownership immediately.
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
        self._lifetime = _ManagerLifetime()
        self.destroyed.connect(self._lifetime.destroy)
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._poll)

    @property
    def busy(self):
        return self.state in ('pending', 'running', 'cancelling', 'committing')

    def _set_state(self, state):
        self.state = state
        if not self._detached and not self._lifetime.dead:
            self.state_changed.emit(state)

    def submit(self, spec):
        if self.busy or self._detached or self._lifetime.dead:
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
        record = _TaskRecord(self.task_id, self._context)
        self._lifetime.record = record
        _MANAGERS[self.task_id] = self
        self._set_state('pending')
        # A state observer may synchronously close/delete the manager.
        if not self._lifetime.dead:
            self._timer.start()
        try:
            Thread(target=_run_task, args=(spec, record), daemon=True).start()
        except BaseException as error:
            with _ACTIVE_LOCK:
                _ACTIVE.discard(spec.key)
            with record.context.lock:
                record.outcome = TaskOutcome(
                    record.task_id, 'failed', error=f'Could not start background worker: {error}'
                )
        return record.task_id

    def _present_progress(self, progress, context):
        if progress is None or self._detached or self._lifetime.dead:
            return
        if context.committing:
            self._set_state('committing')
        elif self.state != 'cancelling':
            self._set_state('running')
        if not self._detached and not self._lifetime.dead:
            self.progress.emit(progress)

    @pyqtSlot()
    def _poll(self):
        if self._lifetime.dead:
            return
        record = self._lifetime.record
        if record is None:
            return
        with record.context.lock:
            progress, record.context.mailbox[0] = record.context.mailbox[0], None
            outcome, record.outcome = record.outcome, None
        try:
            self._present_progress(progress, record.context)
        finally:
            # Even a failing progress observer must not strand task ownership.
            if outcome is not None:
                self._terminal(outcome)

    def _terminal(self, outcome):
        if outcome.task_id != self.task_id:
            return
        try:
            if not self._lifetime.dead:
                self._timer.stop()
            self._set_state(outcome.state)
            if not self._detached and not self._lifetime.dead:
                self.finished.emit(outcome)
        finally:
            # A finished callback can submit a new task. Only release this run;
            # its callback owns the new timer, mailbox, and registry entry.
            _MANAGERS.pop(outcome.task_id, None)

    @property
    def cancellation_requested(self):
        """A late cancel preserves a committed artifact but stops a task pipeline."""
        return self._cancel.is_set()

    def cancel(self):
        if self.busy:
            with self._lock:
                self._cancel.set()
                committing = self._context.committing
            self._set_state('committing' if committing else 'cancelling')

    def detach(self):
        self._detached = True
        self.cancel()
        # Keep polling until terminal cleanup on the GUI thread. If Qt already
        # deleted this manager, its destruction callback released ownership.
        if not self.busy and not self._lifetime.dead:
            self._timer.stop()
