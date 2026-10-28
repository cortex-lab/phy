"""Controller-owned, deferred coordination of exact full-recording summaries."""

from concurrent.futures import CancelledError
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from time import monotonic

from phy.gui.qt import QObject, Qt, QTimer, pyqtSignal, pyqtSlot

from .live_summaries import LiveSummaryEngine, membership_expressions


class _Delivery(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(object)


class _SummaryWorker:
    """One daemon owns pinned, validated source readers across serial jobs."""

    def __init__(self, package_path, source_path, delivery):
        self.package_path = package_path
        self.source_path = source_path
        self.delivery = delivery
        self.stopped = Event()
        self.jobs = Queue(maxsize=1)
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, expressions, cancel):
        self.jobs.put_nowait((expressions, cancel))

    def stop(self):
        self.stopped.set()
        # A queued job observes stopped before opening any source.
        with suppress(Full):
            self.jobs.put_nowait(None)

    def _run(self):
        engine = None
        try:
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                expressions, cancel = job
                results, error = {}, None
                try:
                    if self.stopped.is_set() or cancel.is_set():
                        raise CancelledError()
                    if engine is None:
                        engine = LiveSummaryEngine(self.package_path, source_dir=self.source_path)
                    last_progress = [0.0]

                    def progress(phase, done, total, last_progress=last_progress, cancel=cancel):
                        now = monotonic()
                        if now - last_progress[0] >= 0.25 or done == total:
                            last_progress[0] = now
                            self.delivery.progress.emit((cancel, phase, done, total))

                    results = engine.resolve(expressions, cancel=cancel.is_set, progress=progress)
                except Exception as exc:
                    error = exc
                if self.stopped.is_set():
                    # Terminal delivery means all source handles are released;
                    # a resumed GUI may now create the next worker safely.
                    if engine is not None:
                        engine.close()
                    self.delivery.finished.emit((expressions, results, error, cancel))
                    return
                self.delivery.finished.emit((expressions, results, error, cancel))
        finally:
            if engine is not None:
                engine.close()


class LiveSummaryService(QObject):
    """Keep additive summary primitives outside the curation transaction.

    All public methods run on the owning Qt thread. Construction is suspended;
    ``resume`` starts deferred work. Workers receive immutable membership
    snapshots and own their source handles. Closing never waits for a scan.
    """

    changed = pyqtSignal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self._session = session
        self._package_path = Path(session.path)
        self._source_path = None
        self._engine = LiveSummaryEngine(self._package_path)
        self._expressions = {}
        self._events = None
        self._errors = {}
        self._completed_keys = set()
        self._totals = {}
        self._source_counts = dict(
            zip(session.unit_ids, getattr(session, 'manifest', {}).get('counts', []))
        )
        self._pending = {}
        self._closed = False
        self._suspended = True
        self._running = None
        self._progress_text = None
        self._worker = None
        self._delivery = _Delivery()
        self._delivery.finished.connect(self._completed, Qt.QueuedConnection)
        self._delivery.progress.connect(self._progress, Qt.QueuedConnection)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._synchronize)
        self._refresh()
        self._announced_expressions = self._expressions

    def _refresh(self):
        if self._closed or self._events is self._session.events:
            return False
        expressions = membership_expressions(self._session.unit_ids, self._session.events)
        changed = expressions != self._expressions
        self._expressions = expressions
        self._events = self._session.events
        if changed:
            self._pending = dict(expressions)
            totals = {}
            for expression in expressions.values():
                total = self._expression_total(expression)
                if total is not None:
                    totals[self._engine.key(expression)] = total
            self._totals = totals
            self._completed_keys.intersection_update(
                self._engine.key(expression) for expression in expressions.values()
            )
        return changed

    def _cached(self, expressions):
        try:
            results = self._engine.cached(expressions)
            self._engine.remember(expressions, results)
            self._remember_totals(expressions, results)
            return results
        except Exception as error:
            self._record_error(expressions, error)
            return {}

    def _record_error(self, expressions, error):
        prefix = 'unavailable' if isinstance(error, OSError) else 'error'
        message = f'{prefix}: {error}; retry full summaries or locate the source'
        for expression in expressions.values():
            key = self._engine.key(expression)
            self._errors[key] = message
            if len(self._errors) > 256:
                self._errors.pop(next(iter(self._errors)))

    def _expression_total(self, expression):
        key = self._engine.key(expression)
        if key in self._totals:
            return self._totals[key]
        if not isinstance(expression, dict):
            return None
        if expression['kind'] == 'source':
            value = self._source_counts.get(expression['unit'])
            return None if value is None else int(value)
        if expression['kind'] == 'union':
            parts = [self._expression_total(part) for part in expression['parts']]
            return None if any(part is None for part in parts) else sum(parts)
        return None

    def _remember_totals(self, expressions, results):
        for unit, result in results.items():
            if self._expressions.get(unit) == expressions[unit]:
                key = self._engine.key(expressions[unit])
                self._totals[key] = int(result['counts'].sum())

    def total_count(self, cluster_id):
        """Keep exact scalar counts even when optional native-bin arrays are evicted."""
        if self._closed:
            return None
        self._refresh()
        expression = self._expressions.get(int(cluster_id))
        if expression is None:
            return None
        total = self._expression_total(expression)
        if total is None:
            self.get_summary(cluster_id)
            total = self._expression_total(expression)
        return total

    def get_summary(self, cluster_id):
        """Return readonly primitives for the unit's current expression, if ready."""
        if self._closed:
            return None
        self._refresh()
        cluster_id = int(cluster_id)
        expression = self._expressions.get(cluster_id)
        if expression is None:
            return None
        return self._cached({cluster_id: expression}).get(cluster_id)

    def status(self, cluster_id):
        """Describe missing source evidence without presenting sampled substitutes."""
        if self._closed:
            return 'unavailable: summary service closed'
        if self.get_summary(cluster_id) is not None:
            return 'ready'
        expression = self._expressions.get(int(cluster_id))
        if expression is None:
            return 'unavailable: inactive unit'
        key = self._engine.key(expression)
        if key in self._errors:
            return self._errors[key]
        if key in self._completed_keys:
            return 'unavailable: summary cache budget; retry full summaries'
        if self._suspended:
            return 'cancelled: summary computation suspended'
        if self._running is not None:
            return self._progress_text or 'validating source / recomputing full summaries'
        return 'pending full summary computation'

    def synchronize(self):
        """Coalesce requests until after transaction observers and rollback finish."""
        if not self._closed and not self._timer.isActive():
            self._timer.start(0)

    @pyqtSlot()
    def _synchronize(self):
        if self._closed:
            return
        self._refresh()
        if self._expressions != self._announced_expressions:
            self._announced_expressions = self._expressions
            self.changed.emit()
        if self._suspended or self._running is not None or not self._pending:
            return
        missing = {}
        capacity = max(1, self._engine.max_bytes // self._engine.summary_bytes)
        while self._pending and len(missing) < capacity:
            cluster_id = next(iter(self._pending))
            expression = self._pending.pop(cluster_id)
            if self._engine.key(expression) in self._errors:
                continue
            available = self._cached({cluster_id: expression})
            if not available and self._engine.key(expression) not in self._errors:
                missing[cluster_id] = expression
        if not missing:
            return
        expressions = deepcopy(missing)
        cancel = Event()
        self._running = cancel
        self._progress_text = None
        if self._worker is None:
            self._worker = _SummaryWorker(self._package_path, self._source_path, self._delivery)
        self._worker.submit(expressions, cancel)
        self.changed.emit()

    @pyqtSlot(object)
    def _progress(self, payload):
        cancel, phase, done, total = payload
        if self._closed or self._suspended or cancel is not self._running or cancel.is_set():
            return
        self._progress_text = f'{phase}: {min(100, int(100 * done / max(1, total)))}%'
        self.changed.emit()

    @pyqtSlot(object)
    def _completed(self, payload):
        expressions, results, error, cancel = payload
        if self._closed:
            return
        self._running = None
        self._progress_text = None
        self._refresh()
        # A cancelled/relocated source job must never overwrite a later retry.
        if not cancel.is_set():
            if results:
                self._engine.remember(expressions, results)
                self._remember_totals(expressions, results)
                current_keys = {self._engine.key(e) for e in self._expressions.values()}
                self._completed_keys.update(
                    self._engine.key(expressions[unit]) for unit in results
                )
                self._completed_keys.intersection_update(current_keys)
            if error is not None and not isinstance(error, CancelledError):
                self._record_error(expressions, error)
        self._refresh()
        if not self._suspended:
            self.changed.emit()
            self.synchronize()

    def _stop_worker(self):
        if self._running is not None:
            self._running.set()
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def retry(self, source_path=None):
        """Retry failures, optionally using a relocated immutable source sorting."""
        if self._closed:
            return
        if source_path is not None:
            self._source_path = Path(source_path)
        self._errors.clear()
        self._pending = dict(self._expressions)
        self._stop_worker()
        self.synchronize()
        if not self._suspended:
            self.changed.emit()

    def suspend(self):
        """Cancel work after an accepted GUI close, retaining reusable results."""
        if self._closed:
            return
        self._suspended = True
        self._timer.stop()
        self._stop_worker()

    def resume(self):
        """Resume a newly attached GUI without touching curation persistence."""
        if not self._closed:
            self._suspended = False
            self._pending = dict(self._expressions)
            self.synchronize()

    def close(self):
        """Cancel nonblocking; a late queued result cannot touch session or GUI."""
        if self._closed:
            return
        self.suspend()
        self._closed = True
        self._session = None
        self._timer.stop()
        self._engine.close()
        self._delivery.finished.disconnect(self._completed)
        self._delivery.progress.disconnect(self._progress)
