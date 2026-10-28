"""Deterministic Qt delivery and transaction-boundary summary regressions."""

from concurrent.futures import CancelledError
from threading import Event, get_ident

import numpy as np
import pytest
from phylib.utils import Bunch

from phy.apps.large_recording import summary_service


@pytest.fixture
def controlled_service(qtbot, tmp_path, monkeypatch):
    owner = get_ident()
    calls, engines = [], []
    controls = []

    class Session:
        path = tmp_path
        unit_ids = [0]
        _events = []

        @property
        def events(self):
            assert get_ident() == owner, 'worker accessed live session'
            return self._events

    class Engine:
        max_bytes = 64 * 1024 * 1024
        summary_bytes = 8

        def __init__(self, path, source_dir=None):
            self.cache = {'base': primitive(10)}
            self.source_dir = source_dir
            self.closed = False
            engines.append(self)

        @staticmethod
        def key(expression):
            return expression

        def cached(self, expressions):
            return {c: self.cache[e] for c, e in expressions.items() if e in self.cache}

        def remember(self, expressions, results):
            self.cache.update({expressions[c]: result for c, result in results.items()})

        def resolve(self, expressions, cancel=None, progress=None):
            assert get_ident() != owner
            control = controls[len(calls)]
            calls.append((expressions, self.source_dir))
            control.progress = progress
            control.started.set()
            assert control.release.wait(10), 'test did not release summary worker'
            if cancel():
                raise CancelledError()
            if control.error is not None:
                raise control.error
            return {c: primitive(control.count) for c in expressions}

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def expressions(unit_ids, events):
        result = {0: 'base'}
        for event in events:
            if 'membership' in event:
                result = event['membership'].copy()
        return result

    monkeypatch.setattr(summary_service, 'LiveSummaryEngine', Engine)
    monkeypatch.setattr(summary_service, 'membership_expressions', expressions)
    session = Session()
    service = summary_service.LiveSummaryService(session)

    def control(count=5, error=None):
        item = Bunch(started=Event(), release=Event(), count=count, error=error)
        controls.append(item)
        return item

    fixture = Bunch(
        session=session, service=service, calls=calls, engines=engines, control=control
    )
    try:
        yield fixture
    finally:
        service.close()
        for item in controls:
            item.release.set()
        qtbot.waitUntil(lambda: all(engine.closed for engine in engines))


def primitive(count):
    counts = np.asarray([count], dtype=np.int64)
    counts.flags.writeable = False
    return {'counts': counts, 'trend_sums': {}, 'trend_counts': {}}


def split(fixture, expressions=None):
    fixture.session._events = [
        {'membership': expressions if expressions is not None else {1: 'left', 2: 'right'}}
    ]
    fixture.service.synchronize()


def test_deferred_snapshot_discards_rolled_back_transaction(qtbot, controlled_service):
    f = controlled_service
    changes = []
    f.service.changed.connect(lambda: changes.append(True))
    f.service.resume()
    split(f)
    f.session._events = []  # A later transaction observer rejects the operation.
    qtbot.waitUntil(lambda: not f.service._timer.isActive())
    assert not f.calls
    assert not changes
    assert f.service.get_summary(0)['counts'].sum() == 10


def test_both_children_single_scan_labels_preserve_work(qtbot, controlled_service):
    f = controlled_service
    job = f.control()
    split(f)
    assert f.service.get_summary(1) is None
    assert not f.calls  # Constructor and metric queries cannot start workers.
    f.service.resume()
    qtbot.waitUntil(job.started.is_set)
    assert f.calls[0][0] == {1: 'left', 2: 'right'}
    changes = []
    f.service.changed.connect(lambda: changes.append(get_ident()))
    f.session._events = f.session.events + [{'operation': 'label'}]
    f.service.synchronize()
    qtbot.waitUntil(lambda: not f.service._timer.isActive())
    assert not changes
    assert len(f.calls) == 1
    assert 'recomputing' in f.service.status(1)
    job.release.set()
    qtbot.waitUntil(lambda: f.service.get_summary(1) is not None)
    assert f.service.get_summary(2)['counts'].sum() == 5
    assert all(thread == get_ident() for thread in changes)
    assert not f.service.get_summary(1)['counts'].flags.writeable


def test_branch_completion_cached_but_never_attached_to_reused_id(qtbot, controlled_service):
    f = controlled_service
    first, second = f.control(3), f.control(7)
    f.service.resume()
    split(f)
    qtbot.waitUntil(first.started.is_set)
    split(f, {1: 'different-left', 2: 'different-right'})
    assert f.service.get_summary(1) is None
    first.release.set()
    qtbot.waitUntil(second.started.is_set)
    assert f.service.get_summary(1) is None
    second.release.set()
    qtbot.waitUntil(lambda: f.service.get_summary(1) is not None)
    assert f.service.get_summary(1)['counts'].sum() == 7
    split(f)  # Undo to the earlier branch: its result remains reusable.
    qtbot.waitUntil(lambda: not f.service._timer.isActive())
    assert f.service.get_summary(1)['counts'].sum() == 3
    assert len(f.calls) == 2


def test_undo_while_running_and_redo_reuses_result(qtbot, controlled_service):
    f = controlled_service
    job = f.control()
    f.service.resume()
    split(f)
    qtbot.waitUntil(job.started.is_set)
    f.session._events = []
    f.service.synchronize()
    job.release.set()
    qtbot.waitUntil(lambda: f.service._running is None)
    assert f.service.get_summary(1) is None
    assert f.service.get_summary(0)['counts'].sum() == 10
    split(f)
    qtbot.waitUntil(lambda: not f.service._timer.isActive())
    assert f.service.get_summary(1)['counts'].sum() == 5
    assert len(f.calls) == 1


def test_source_failure_requires_explicit_retry_and_relocation(
    qtbot, controlled_service, tmp_path
):
    f = controlled_service
    first = f.control(error=FileNotFoundError('source moved'))
    second = f.control(8)
    f.service.resume()
    split(f)
    qtbot.waitUntil(first.started.is_set)
    first.release.set()
    qtbot.waitUntil(lambda: 'source moved' in f.service.status(1))
    assert 'unavailable' in f.service.status(1)
    f.service.synchronize()
    qtbot.waitUntil(lambda: not f.service._timer.isActive())
    assert len(f.calls) == 1
    relocated = tmp_path / 'relocated'
    f.service.retry(source_path=relocated)
    qtbot.waitUntil(second.started.is_set)
    assert f.calls[1][1] == relocated
    second.release.set()
    qtbot.waitUntil(lambda: f.service.get_summary(1) is not None)
    assert f.service.status(1) == 'ready'


def test_close_nonblocking_drops_queued_delivery(qtbot, controlled_service):
    f = controlled_service
    job = f.control()
    f.service.resume()
    split(f)
    qtbot.waitUntil(job.started.is_set)
    changes = []
    f.service.changed.connect(lambda: changes.append(True))
    f.service.close()  # Worker is still blocked; cannot wait for it here.
    assert not job.release.is_set()
    assert f.service._running.is_set()
    assert f.service.get_summary(1) is None
    job.release.set()
    qtbot.waitUntil(lambda: all(engine.closed for engine in f.engines))
    assert not changes
    assert 'closed' in f.service.status(1)


def test_suspend_resume_serializes_cancelled_worker(qtbot, controlled_service):
    f = controlled_service
    first, second = f.control(), f.control(9)
    f.service.resume()
    split(f)
    qtbot.waitUntil(first.started.is_set)
    f.service.suspend()
    assert 'cancelled' in f.service.status(1)
    f.service.resume()
    qtbot.waitUntil(lambda: not f.service._timer.isActive())
    assert len(f.calls) == 1
    first.release.set()
    qtbot.waitUntil(second.started.is_set)
    second.release.set()
    qtbot.waitUntil(lambda: f.service.get_summary(1) is not None)
    assert f.service.get_summary(1)['counts'].sum() == 9
    assert f.engines[1].closed
    assert not f.engines[-1].closed
    f.service.suspend()
    qtbot.waitUntil(lambda: all(engine.closed for engine in f.engines[1:]))


def test_successive_jobs_reuse_worker_owned_engine(qtbot, controlled_service):
    f = controlled_service
    first, second = f.control(), f.control(9)
    f.service.resume()
    split(f)
    qtbot.waitUntil(first.started.is_set)
    first.release.set()
    qtbot.waitUntil(lambda: f.service.get_summary(1) is not None)
    split(f, {3: 'nested-left', 4: 'nested-right'})
    qtbot.waitUntil(second.started.is_set)
    assert len(f.engines) == 2  # One GUI resolver, one persistent worker reader.
    assert not f.engines[1].closed
    second.release.set()
    qtbot.waitUntil(lambda: f.service.get_summary(3) is not None)


def test_cache_error_returns_explicit_status(qtbot, controlled_service, monkeypatch):
    f = controlled_service

    def too_large(expressions):
        raise ValueError('Live summary output budget exceeded.')

    monkeypatch.setattr(f.service._engine, 'cached', too_large)
    assert f.service.get_summary(0) is None
    assert 'budget exceeded' in f.service.status(0)
    f.service.resume()
    qtbot.waitUntil(lambda: not f.service._timer.isActive())
    assert not f.calls


def test_aggregate_requests_are_batched_to_output_budget(qtbot, controlled_service):
    f = controlled_service
    f.service._engine.max_bytes = 16  # Two eight-byte summaries per job.
    first, second = f.control(), f.control(9)
    f.service.resume()
    split(f, {1: 'a', 2: 'b', 3: 'c', 4: 'd'})
    qtbot.waitUntil(first.started.is_set)
    assert f.calls[0][0] == {1: 'a', 2: 'b'}
    first.release.set()
    qtbot.waitUntil(second.started.is_set)
    assert f.calls[1][0] == {3: 'c', 4: 'd'}
    second.release.set()
    qtbot.waitUntil(lambda: f.service.get_summary(4) is not None)
    assert len(f.calls) == 2


def test_real_worker_exact_primitives_merge_offline_and_no_curation_side_effects(qtbot, tmp_path):
    from copy import deepcopy

    from phy.apps.large_recording.curation import BoundedSession
    from phy.apps.large_recording.prepare import prepare_large
    from phy.apps.large_recording.tests.test_source import make_sorting

    params = make_sorting(tmp_path / 'source')
    bounded = prepare_large(
        params, tmp_path / 'bounded', window_seconds=3, stratum_seconds=20, bin_seconds=5
    )
    session = BoundedSession(bounded)
    predicate = {
        'attribute': 'amplitudes.npy',
        'operator': '>=',
        'threshold': 4,
        'nonfinite': 'keep',
    }
    session.record_split(0, 100, 101, predicate)
    events, metadata = deepcopy(session.events), deepcopy(session.metadata)
    assignments = session.assignments.copy()
    original_files = {p: p.read_bytes() for p in bounded.iterdir() if p.is_file()}
    service = summary_service.LiveSummaryService(session)
    try:
        service.resume()
        qtbot.waitUntil(lambda: service.get_summary(100) is not None)
        np.testing.assert_array_equal(service.get_summary(100)['counts'], [0, 0, 1, 1])
        np.testing.assert_array_equal(service.get_summary(101)['counts'], [2, 0, 0, 0])
        np.testing.assert_array_equal(
            service.get_summary(100)['trend_sums']['amplitude'], [0, 0, 6, 9]
        )
        assert session.events == events
        assert session.metadata == metadata
        np.testing.assert_array_equal(session.assignments, assignments)
        assert {p: p.read_bytes() for p in original_files} == original_files
        worker = service._worker
        service.suspend()
        qtbot.waitUntil(lambda: not worker.thread.is_alive())
        params.parent.rename(tmp_path / 'source-offline')
        session.record_merge([100, 1], 102)
        service.synchronize()
        np.testing.assert_array_equal(service.get_summary(102)['counts'], [1, 1, 2, 1])
        assert service.status(102) == 'ready'
        assert service.total_count(102) == 5
        assert len(service._totals) <= len(service._expressions)
    finally:
        service.close()
        session.close()


def test_evicted_rows_do_not_restart_scans_on_table_redraw(qtbot, controlled_service, monkeypatch):
    f = controlled_service
    engine = f.service._engine
    engine.max_bytes = 16

    def bounded_remember(expressions, results):
        for unit, result in results.items():
            engine.cache[expressions[unit]] = result
            while len(engine.cache) > 2:
                engine.cache.pop(next(iter(engine.cache)))

    monkeypatch.setattr(engine, 'remember', bounded_remember)
    first, second = f.control(), f.control(9)
    # Emulate the table refreshing every unit on each completion, with no disk
    # cache available to restore rows evicted by the tiny resident budget.
    f.service.changed.connect(lambda: [f.service.get_summary(unit) for unit in range(1, 5)])
    f.service.resume()
    split(f, {1: 'a', 2: 'b', 3: 'c', 4: 'd'})
    qtbot.waitUntil(first.started.is_set)
    first.release.set()
    qtbot.waitUntil(second.started.is_set)
    second.release.set()
    qtbot.waitUntil(lambda: f.service._running is None and not f.service._timer.isActive())
    assert len(f.calls) == 2
    assert f.service.get_summary(1) is None
    assert 'cache budget' in f.service.status(1)
    assert f.service.total_count(1) == 5
    assert f.service.total_count(4) == 9
    assert len(f.calls) == 2


def test_progress_is_queued_and_stale_or_cancelled_updates_are_ignored(qtbot, controlled_service):
    f = controlled_service
    job = f.control()
    f.service.resume()
    split(f)
    qtbot.waitUntil(job.started.is_set)
    job.progress('reading prepared unit summaries', 1, 2)
    qtbot.waitUntil(lambda: '50%' in f.service.status(1))
    running = f.service._running
    f.service._progress((Event(), 'stale', 1, 1))
    assert '50%' in f.service.status(1)
    running.set()
    f.service._progress((running, 'cancelled', 1, 1))
    assert 'cancelled' not in f.service._progress_text
    job.release.set()
    qtbot.waitUntil(lambda: f.service._running is None)
