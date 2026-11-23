"""Selected-visible RP work with isolated, coalesced Qt delivery."""

import hashlib
from collections import OrderedDict
from concurrent.futures import CancelledError
from contextlib import suppress
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from time import monotonic

import numpy as np

from phy.gui.qt import QObject, Qt, QTimer, pyqtSignal
from phy.gui.tasks import heavy_io

from .live_summaries import (
    _check_cancel,
    _signature,
    acquire_source_engine,
    release_source_engine,
)
from .refractory import RefractoryAccumulator, RefractoryUnavailable
from .refractory_engine import RefractoryEngine
from .source import Source, hash_json


class _Delivery(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(object)


class _Worker:
    def __init__(self, path, bounded, source_path, delivery):
        self.path, self.bounded, self.source_path = path, bounded, source_path
        self.delivery = delivery
        self.stopped = Event()
        self.jobs = Queue(maxsize=1)
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        with suppress(Full):
            self.jobs.put_nowait(None)

    def run(self):
        lease = engine = source = None
        signatures, cache = {}, OrderedDict()
        try:
            while not self.stopped.is_set():
                job = self.jobs.get()
                if job is None:
                    break
                payload, cancel = job
                error, results = None, {}
                last_progress = [0.0]

                def cancelled(cancel=cancel):
                    return self.stopped.is_set() or cancel.is_set()

                def check():
                    _check_cancel(cancelled)

                def progress(phase, done, total, cancel=cancel, last_progress=last_progress):
                    now = monotonic()
                    if now - last_progress[0] >= 0.25 or done == total:
                        last_progress[0] = now
                        self.delivery.progress.emit((cancel, phase, done, total))

                try:
                    with heavy_io(
                        check, waiting=lambda: progress('waiting for recording I/O', 0, 0)
                    ):
                        check()
                        if self.bounded:
                            if engine is None:
                                lease, reader = acquire_source_engine(self.path, self.source_path)
                                engine = RefractoryEngine(reader)
                            results = engine.resolve(payload, cancelled, progress)
                        else:
                            if source is None:
                                source = Source(
                                    self.path / 'params.py', cancel=cancelled, progress=progress
                                )
                                names = ['spike_times.npy', 'amplitudes.npy', 'params.py']
                                names += [
                                    n
                                    for n in (
                                        '_phy_spikes_subset.manifest.json',
                                        'timestamps.json',
                                    )
                                    if (self.path / n).exists()
                                ]
                                signatures = {n: _signature((self.path / n).stat()) for n in names}

                            def unchanged(signatures=signatures):
                                if any(
                                    _signature((self.path / n).stat()) != sig
                                    for n, sig in signatures.items()
                                ):
                                    raise ValueError(
                                        'Original timing, amplitudes or exposure changed; reopen the view.'
                                    )

                            unchanged()
                            pending_cache = {}
                            for unit, ids in payload.items():
                                check()
                                # Worker-only canonicalization; membership arrays are frozen snapshots.
                                digest = hashlib.sha256()
                                for start in range(0, len(ids), 100000):
                                    check()
                                    digest.update(
                                        memoryview(
                                            np.ascontiguousarray(ids[start : start + 100000])
                                        ).cast('B')
                                    )
                                key = hash_json(
                                    {
                                        'rows': digest.hexdigest(),
                                        'dtype': str(ids.dtype),
                                        'count': len(ids),
                                        'source': source.identity['sha256'],
                                    }
                                )
                                if key in cache:
                                    value = cache[key]
                                else:
                                    try:
                                        if 'amplitudes.npy' not in source.arrays:
                                            raise RefractoryUnavailable(
                                                'Raw source amplitudes unavailable.'
                                            )
                                        acc = RefractoryAccumulator(
                                            source.sample_rate,
                                            source.num_samples,
                                            source.valid_intervals,
                                        )
                                        for start in range(0, len(ids), 100000):
                                            check()
                                            rows = ids[start : start + 100000]
                                            acc.feed(
                                                source.arrays['spike_times.npy'].reshape(-1)[rows],
                                                source.arrays['amplitudes.npy'].reshape(-1)[rows],
                                                cancelled,
                                            )
                                        value = acc.finish(cancelled)
                                    except RefractoryUnavailable as exc:
                                        value = {'status': 'unavailable', 'message': str(exc)}
                                    pending_cache[key] = value
                                results[unit] = value
                            unchanged()
                            check()
                            cache.update(pending_cache)
                            while (
                                len(cache) > 128
                                or sum(
                                    v.nbytes
                                    for r in cache.values()
                                    for v in r.values()
                                    if isinstance(v, np.ndarray)
                                )
                                > 8388608
                            ):
                                cache.popitem(last=False)
                        check()
                except Exception as exc:
                    error = exc
                    results = {}
                    if not isinstance(exc, CancelledError):
                        cache.clear()
                self.delivery.finished.emit((cancel, results, error))
        finally:
            if lease is not None:
                release_source_engine(lease)
            if source is not None:
                source.close()


class RefractoryService(QObject):
    """One current demand, one cancellable job; no session/model reaches workers."""

    changed = pyqtSignal()

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.results, self.message = {}, 'Select a unit.'
        self._token = self._payload = self._running = self._worker = None
        self._membership_references = ()
        self._closed = False
        self._delivery = _Delivery()
        self._delivery.finished.connect(self._completed, Qt.QueuedConnection)
        self._delivery.progress.connect(self._progress, Qt.QueuedConnection)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start)

    def request(self, ids, enabled=True):
        if self._closed:
            return
        ids = tuple(ids) if enabled else ()
        session = getattr(self.controller, 'bounded_session', None)
        if session is not None:
            payload = (
                session.membership_snapshot(ids) if hasattr(session, 'membership_snapshot') else {}
            )
            references = tuple(payload.values())
        else:
            supervisor = getattr(self.controller, 'supervisor', None)
            payload = (
                supervisor.clustering.membership_snapshot(ids) if supervisor is not None else {}
            )
            references = tuple(supervisor.clustering.spikes_per_cluster[u] for u in payload)
        token = (ids, tuple((u, id(value)) for u, value in zip(payload, references, strict=True)))
        if token == self._token:
            return
        self._token, self._payload = token, payload
        self._membership_references = references
        self.results = {}
        self.message = (
            'RP pending: validating original evidence.'
            if payload
            else ('Select a unit.' if not ids else 'RP evidence unavailable.')
        )
        if self._running is not None:
            self._running.set()
        self._timer.start(0)

    def _start(self):
        if self._closed or self._running is not None or not self._payload:
            return
        path = getattr(self.controller, 'dir_path', None)
        if path is None:
            self.message = 'RP evidence unavailable: original source path missing.'
            self.changed.emit()
            return
        if self._worker is None:
            session = getattr(self.controller, 'bounded_session', None)
            summary = getattr(self.controller, '_live_summaries', None)
            self._worker = _Worker(
                Path(path),
                session is not None,
                getattr(summary, '_source_path', None),
                self._delivery,
            )
        self._running = Event()
        self._worker.jobs.put_nowait((self._payload, self._running))

    def _completed(self, delivery):
        cancel, results, error = delivery
        if self._closed or cancel is not self._running:
            return
        self._running = None
        if cancel.is_set():
            self._timer.start(0)
            return
        self.results = results if error is None else {}
        self.message = f'RP unavailable: {error}' if error is not None else ''
        self.changed.emit()

    def _progress(self, delivery):
        cancel, phase, done, total = delivery
        if self._closed or cancel is not self._running or cancel.is_set():
            return
        self.message = f'RP {phase}' + (f' ({done / total:.0%})' if total else '')
        self.changed.emit()

    def close(self):
        self._closed = True
        self._timer.stop()
        self.results.clear()
        self._payload = None
        self._membership_references = ()
        if self._running is not None:
            self._running.set()
        if self._worker is not None:
            self._worker.stop()
