"""Coalesced background draft protection at settled bounded curation boundaries."""

import copy
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path

from phylib.utils import emit

from phy.gui.qt import QObject, Qt, QTimer, pyqtSignal

from ._immutable import freeze
from .recovery import (
    FORMAT,
    VERSION,
    RecoveryStore,
    build_recovered_state,
    default_recovery_root,
    recovery_bindings,
)

logger = logging.getLogger(__name__)


class _Delivery(QObject):
    finished = pyqtSignal(object)


class _Writer:
    """One active operation and one latest pending snapshot; controls stay ordered."""

    def __init__(self, factory, delivery):
        self.factory, self.delivery = factory, delivery
        self.condition = threading.Condition()
        self.pending = None
        self.controls = deque()
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name='phy-recovery', daemon=True)
        self.thread.start()

    def submit(self, kind, revision, operation, snapshot=False):
        future = Future()
        with self.condition:
            if self.stopping:
                future.set_exception(RuntimeError('Recovery writer is closed.'))
                return future
            item = (kind, revision, operation, future)
            if snapshot:
                if self.pending is not None:
                    self.pending[-1].cancel()
                self.pending = item
            else:
                self.controls.append(item)
            self.condition.notify()
        return future

    def stop(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()

    def _deliver(self, kind, revision, future):
        # GUI/interpreter shutdown never changes a completed disk write.
        with suppress(RuntimeError):
            self.delivery.finished.emit((kind, revision, future))

    def _run(self):
        store = None
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.controls or self.pending or self.stopping)
                    if self.controls:
                        item = self.controls.popleft()
                    elif self.pending is not None:
                        item, self.pending = self.pending, None
                    elif self.stopping:
                        return
                    else:
                        continue
                kind, revision, operation, future = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    if store is None:
                        store = self.factory()
                    result = operation(store)
                except Exception as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
                self._deliver(kind, revision, future)
        finally:
            if store is not None:
                store.close()


class RecoveryCoordinator(QObject):
    """Expose protection acknowledgements separately from canonical Save state."""

    changed = pyqtSignal()

    def __init__(self, controller, root=None, enabled=True, idle_ms=1000, maximum_ms=5000):
        super().__init__()
        self.controller = controller
        self.supervisor = controller.supervisor
        self.session = controller.bounded_session
        self.root = Path(root) if root is not None else default_recovery_root()
        self.enabled = bool(enabled)
        self.session_id = uuid.uuid4().hex
        self.bindings = recovery_bindings(self.session)
        self.idle_ms, self.maximum_ms = idle_ms, maximum_ms
        self._last_token = self.supervisor.recovery_token()
        self._current_revision = self._protected_revision = self._checkpoint_revision = 0
        self._last_protected_at = None
        self._last_checkpoint_at = (
            (self.session.checkpoint_path / 'checkpoint.json').stat().st_mtime
            if self.session.checkpoint_path is not None
            else None
        )
        self._state = (
            'protected'
            if self.enabled and self._last_checkpoint_at is not None
            else 'ready'
            if self.enabled
            else 'disabled'
        )
        self._error = self._warning = None
        self._error_revision = None
        self._writer = None
        self._closed = self._started = self._retired = False
        self._retry_ms = 1000
        self._restored_from = None
        self.pending_drafts = []
        self.startup_error = None
        self._delivery = _Delivery()
        self._delivery.finished.connect(self._completed, Qt.QueuedConnection)
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.timeout.connect(self.request_flush)
        self._deadline = QTimer(self)
        self._deadline.setSingleShot(True)
        self._deadline.timeout.connect(self.request_flush)

    @property
    def status(self):
        return {
            'enabled': self.enabled,
            'state': self._state,
            'current_revision': self._current_revision,
            'protected_revision': self._protected_revision,
            'last_protected_at': self._last_protected_at,
            'last_checkpoint_at': self._last_checkpoint_at,
            'error': self._error,
            'warning': self._warning,
            'startup_error': self.startup_error,
            'recovery_path': str(self.root) if self.root is not None else None,
        }

    def _announce(self):
        self.changed.emit()
        emit('recovery_status_changed', self.supervisor)

    def _ensure_writer(self):
        if self._writer is None:
            session, root, session_id = self.session, self.root, self.session_id
            self._writer = _Writer(
                lambda: RecoveryStore(session, root=root, session_id=session_id), self._delivery
            )
        return self._writer

    def start(self):
        if not self.enabled or self._closed:
            return
        self._started = True
        self._ensure_writer()
        if self._retired and self.supervisor.is_dirty():
            self.observe(force=True)
        elif self._current_revision > self._protected_revision:
            self._schedule()

    def suspend(self):
        if self.enabled and not self._closed and self._started:
            self.request_flush()
        self._started = False
        self._idle.stop()
        self._deadline.stop()

    def observe(self, force=False):
        if not self.enabled or self._closed:
            return
        token = self.supervisor.recovery_token()
        if not force and token == self._last_token:
            return
        self._last_token = token
        self._current_revision += 1
        self._retired = False
        self._state = 'pending'
        if self._started:
            self._schedule()
        self._announce()

    def _schedule(self):
        self._idle.start(self.idle_ms)
        if not self._deadline.isActive():
            self._deadline.start(self.maximum_ms)

    def _snapshot(self):
        # All references are immutable. JSON encoding and ledger materialization
        # belong to the worker, never to an interactive spike-array transaction.
        events, allocator, _ = self.supervisor.recovery_token()
        catalog = (
            self.supervisor.merge_propositions.catalog
            if self.supervisor.merge_propositions is not None
            else None
        )
        ledger = freeze(self.session.extra_documents.get('curation_review.json'))
        base = copy.deepcopy(self.session._expected_pointer)
        revision, created = self._current_revision, time.time()
        source_hash = self.session.proposition_documents.get('curation.json', {}).get('sha256')
        session_id, bindings, path_hint = self.session_id, self.bindings, str(self.session.path)

        def materialize(store):
            reviews = catalog.review_mapping(source_sha256=source_hash) if catalog else ledger
            payload = {
                'format': FORMAT,
                'version': VERSION,
                'session_id': session_id,
                'revision': revision,
                'created_at': created,
                'bindings': bindings,
                'base_pointer': base,
                'dataset_path_hint': path_hint,
                'events': list(events),
                'next_cluster_id': allocator,
                'reviews': reviews,
            }
            result = store.write(payload)
            return {**result, 'protected_at': time.time(), 'base_pointer': base}

        return revision, materialize

    def request_flush(self):
        if not self.enabled or self._closed or self._retired:
            return None
        if getattr(self.supervisor, '_recovery_depth', 0):
            if self._started:
                self._idle.start(self.idle_ms)
            return None
        self.observe()
        if self._current_revision <= self._protected_revision:
            if self._state == 'error':
                return self._ensure_writer().submit(
                    'probe', self._current_revision, lambda store: store.probe()
                )
            return None
        self._idle.stop()
        self._deadline.stop()
        revision, operation = self._snapshot()
        self._state = 'saving'
        self._announce()
        return self._ensure_writer().submit('snapshot', revision, operation, snapshot=True)

    def _completed(self, payload):
        kind, revision, future = payload
        if self._closed or future.cancelled():
            return
        if kind == 'snapshot' and revision <= max(
            self._checkpoint_revision, self._protected_revision
        ):
            return  # An explicit durable Save already protects this older work.
        try:
            result = future.result()
        except Exception as error:
            if kind == 'checkpoint':
                self._warning = f'Checkpoint saved, but recovery retirement failed: {error}'
                self._announce()
                return
            self._error = str(error)
            self._error_revision = revision
            self._state = 'error'
            logger.warning('Automatic recovery failed: %s', error)
            if self._started:
                self._idle.start(self._retry_ms)
                self._retry_ms = min(60000, self._retry_ms * 2)
        else:
            if kind == 'snapshot' and not result.get('retired'):
                if result['base_pointer'] != self.session._expected_pointer:
                    self._state = 'pending'
                    if self._started:
                        self._schedule()
                    self._announce()
                    return
                if result.get('checkpoint_conflict'):
                    self._error = (
                        'Another session saved a different checkpoint. This recovery draft '
                        'was retained separately; automatic rebasing is unavailable.'
                    )
                    self._error_revision = revision
                    self._state = 'error'
                    self._announce()
                    return
                self._protected_revision = max(self._protected_revision, revision)
                self._last_protected_at = result['protected_at']
                self._warning = result.get('warning')
                if self._error_revision is None or revision >= self._error_revision:
                    self._error = None
                    self._error_revision = None
                    self._retry_ms = 1000
                    self._state = (
                        'protected'
                        if self._protected_revision >= self._current_revision
                        else 'pending'
                    )
            elif kind in ('checkpoint', 'retire') and result and result.get('warning'):
                self._warning = result['warning']
            elif kind == 'probe':
                self._error = None
                self._error_revision = None
                self.startup_error = None
                self._state = (
                    'protected'
                    if self._last_protected_at is not None or self._last_checkpoint_at is not None
                    else 'ready'
                )
            elif kind == 'discard':
                if result and result.get('warning'):
                    self._warning = result['warning']
                self._retired = revision >= self._current_revision
                self._state = 'pending'
                self._protected_revision = 0
                self._last_protected_at = None
                self._error = None
                if not self._retired and self._started:
                    self._schedule()
        self._announce()

    def checkpoint_saved(self, token):
        if not self.enabled or self._closed:
            self._last_checkpoint_at = time.time()
            if not self._closed:
                self._state = 'disabled'
                self._announce()
            return
        self.observe()
        # Save's captured state can differ from a newer edit. Retire only its
        # acknowledged prefix; newer queued snapshots must survive.
        revision = token[0]
        self._checkpoint_revision = max(self._checkpoint_revision, revision)
        self._last_checkpoint_at = time.time()
        restored = self._restored_from

        def retire_checkpoint(store):
            results = [store.retire(revision, reason='checkpointed')]
            if restored is not None:
                results.append(
                    store.retire(
                        restored['payload']['revision'], 'checkpointed', restored['branch']
                    )
                )
            return {
                'warning': '; '.join(r['warning'] for r in results if r.get('warning')) or None
            }

        self._ensure_writer().submit('checkpoint', revision, retire_checkpoint)
        if self.supervisor.recovery_token() == token[1]:
            self._protected_revision = self._current_revision
            self._last_protected_at = self._last_checkpoint_at
            self._state = 'protected'
            self._error = None
        else:
            self._state = 'pending'
            if self._started:
                self._schedule()
        self._announce()

    def checkpoint_failed(self, error):
        self.observe()
        self._state = 'error'
        self._error = f'Checkpoint save failed or durability is uncertain: {error}'
        self._announce()

    def checkpoint_token(self):
        self.observe()
        return self._current_revision, self.supervisor.recovery_token()

    def discard(self):
        if not self.enabled or self._closed:
            future = Future()
            future.set_result(None)
            return future
        self.observe()
        self._idle.stop()
        self._deadline.stop()
        revision = self._current_revision
        restored = self._restored_from

        def retire(store):
            results = [store.retire(revision, reason='discarded')]
            if restored is not None:
                results.append(
                    store.retire(restored['payload']['revision'], 'discarded', restored['branch'])
                )
            return {
                'warning': '; '.join(r['warning'] for r in results if r.get('warning')) or None
            }

        return self._ensure_writer().submit('discard', revision, retire)

    def candidates(self):
        base = copy.deepcopy(self.session._expected_pointer)
        return self._ensure_writer().submit('candidates', 0, lambda store: store.candidates(base))

    def restore(self, candidate):
        session, payload = self.session, candidate['payload']
        return self._ensure_writer().submit(
            'restore', 0, lambda store: build_recovered_state(session, payload)
        )

    def retire_candidate(self, candidate):
        return self._ensure_writer().submit(
            'retire',
            0,
            lambda store: store.retire(
                candidate['payload']['revision'], 'discarded', candidate['branch']
            ),
        )

    def offer_startup(self):
        if not self.enabled or self._closed:
            return
        try:
            drafts, warnings = wait_operation(self.candidates(), 'Checking recovery drafts…')
            self.pending_drafts = drafts
            if warnings:
                self.startup_error = '; '.join(dict.fromkeys(warnings))
                self._announce()
            if not drafts:
                return
            choice = choose_recovery(drafts)
            action, index = (choice, 0) if isinstance(choice, str) else choice
            candidate = drafts[index]
            if action == 'restore':
                recovered = wait_operation(
                    self.restore(candidate), 'Reconstructing recovery draft…'
                )
                self._restored_from = candidate
                self.restored(recovered)
            elif action == 'discard':
                wait_operation(self.retire_candidate(candidate), 'Discarding recovery draft…')
        except Exception as error:
            self.startup_error = str(error)
            self._error = f'Recovery could not be restored: {error}'
            self._state = 'error'
            self._announce()

    def restored(self, recovered):
        self.supervisor.adopt_recovery(recovered)
        for name in ('_lineage_cache', '_waveform_membership_cache'):
            if hasattr(self.controller, name):
                delattr(self.controller, name)
        self.observe(force=True)
        self.request_flush()

    def close(self):
        if self._closed:
            return
        self.suspend()
        self._closed = True
        if self._writer is not None:
            self._writer.stop()


def wait_operation(future, title, parent=None):
    """Wait with an active Qt event loop, never doing disk/reconstruction work here."""
    from PyQt5.QtWidgets import QProgressDialog

    from phy.gui.qt import QEventLoop

    if future.done():
        return future.result()
    progress = QProgressDialog(title, '', 0, 0, parent)
    progress.setCancelButton(None)
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setMinimumDuration(100)
    progress.show()
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if future.done() else None)
    timer.start()
    loop.exec_()
    timer.stop()
    progress.close()
    return future.result()


from phy.gui.qt import mockable


@mockable
def choose_recovery(candidates):  # pragma: no cover -- exercised through mockable choice
    from datetime import datetime

    from PyQt5.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

    dialog = QDialog()
    dialog.setWindowTitle('Recover unsaved bounded curation')
    layout = QVBoxLayout(dialog)
    layout.addWidget(
        QLabel(
            'A recovery draft contains unsaved assignments, labels and review decisions.\n'
            'Restoring it does not change the saved checkpoint or full-source replay.\n'
            'The restored draft starts a new Undo baseline.'
        )
    )
    choices = QComboBox()
    for candidate in candidates:
        payload = candidate['payload']
        stamp = (
            datetime.fromtimestamp(payload['created_at'])
            .astimezone()
            .strftime('%Y-%m-%d %H:%M:%S %Z')
        )
        choices.addItem(f'{stamp} — {len(payload["events"])} journal events')
    layout.addWidget(choices)
    buttons = QDialogButtonBox()
    restore = buttons.addButton('Restore draft', QDialogButtonBox.AcceptRole)
    discard = buttons.addButton(
        'Keep checkpoint and discard draft', QDialogButtonBox.DestructiveRole
    )
    later = buttons.addButton('Decide later', QDialogButtonBox.RejectRole)
    selection = ['later']
    restore.clicked.connect(lambda: (selection.__setitem__(0, 'restore'), dialog.accept()))
    discard.clicked.connect(lambda: (selection.__setitem__(0, 'discard'), dialog.accept()))
    later.clicked.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec_()
    return selection[0], choices.currentIndex()
