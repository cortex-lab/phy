"""Demand gating, stale IDs, worker delivery and persistent-artist regressions."""

from queue import Queue
from threading import Event

import numpy as np
from phylib.utils import Bunch

from phy.apps.large_recording import refractory_service
from phy.apps.large_recording.curation import BoundedSession
from phy.apps.large_recording.tests.test_refractory_engine import make_package


def test_demand_coalescing_stale_completion_reused_id_and_close(qtbot, tmp_path, monkeypatch):
    source, bounded = make_package(tmp_path)
    workers = []

    class Worker:
        def __init__(self, *args):
            self.jobs = Queue()
            self.stopped = False
            workers.append(self)

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(refractory_service, '_Worker', Worker)
    with BoundedSession(bounded) as session:
        service = refractory_service.RefractoryService(
            Bunch(dir_path=bounded, bounded_session=session)
        )
        service.request([0], enabled=False)
        service._start()
        assert not workers
        service.request([0])
        service._start()
        payload, first = workers[0].jobs.get_nowait()
        service.request([0])
        assert not first.is_set()  # Unchanged membership and metadata reuse work.
        original = session.snapshot()
        session.record_merge([0, 1], 100)
        service.request([100])
        assert first.is_set()
        service._completed((first, {0: {'old': True}}, None))
        assert not service.results
        service._start()
        _, second = workers[0].jobs.get_nowait()
        session.restore(original)
        session.record_merge([0, 2], 100)
        service.request([100])
        assert second.is_set()
        service._completed((second, {100: {'old': True}}, None))
        service._start()
        _, third = workers[0].jobs.get_nowait()
        service._completed((third, {100: {'current': True}}, None))
        assert service.results[100]['current']
        service.close()
        assert workers[0].stopped
        service._completed((third, {100: {'late': True}}, None))
        assert not service.results


def test_actual_worker_current_membership_delivery_and_hidden_cancel(qtbot, tmp_path):
    _, bounded = make_package(tmp_path)
    with BoundedSession(bounded) as session:
        service = refractory_service.RefractoryService(
            Bunch(dir_path=bounded, bounded_session=session)
        )
        try:
            service.request([0, 1])
            qtbot.waitUntil(lambda: bool(service.results), timeout=10000)
            assert set(service.results) == {0, 1}
            session.record_merge([0, 1], 100)
            service.request([100])
            assert not service.results
            qtbot.waitUntil(lambda: 100 in service.results, timeout=10000)
            assert service.results[100]['duplicates_removed'] > 0
            service.request([], enabled=False)
            assert not service.results
        finally:
            worker = service._worker
            service.close()
            qtbot.waitUntil(lambda: not worker.thread.is_alive(), timeout=10000)


def test_ordinary_worker_uses_frozen_original_rows(qtbot, tmp_path):
    source, _ = make_package(tmp_path)
    from phy.cluster.clustering import Clustering

    clustering = Clustering(np.load(source / 'spike_clusters.npy'))
    snapshot = clustering.membership_snapshot([0, 1])
    assert not snapshot[0].flags.writeable
    service = refractory_service.RefractoryService(
        Bunch(
            dir_path=source,
            supervisor=Bunch(clustering=clustering),
        )
    )
    try:
        service.request([0])
        qtbot.waitUntil(
            lambda: bool(service.results) or service.message.startswith('RP unavailable:'),
            timeout=10000,
        )
        assert 0 in service.results, service.message
        old = snapshot[0].copy()
        clustering.merge([0, 1])
        np.testing.assert_array_equal(snapshot[0], old)
        target = max(clustering.cluster_ids)
        service.request([target])
        qtbot.waitUntil(lambda: target in service.results, timeout=10000)
        assert service.results[target]['duplicates_removed'] > 0
    finally:
        worker = service._worker
        service.close()
        if worker is not None:
            qtbot.waitUntil(lambda: not worker.thread.is_alive(), timeout=10000)


def test_unrelated_late_progress_does_not_publish(qtbot, tmp_path):
    service = refractory_service.RefractoryService(Bunch(dir_path=tmp_path))
    service._running = Event()
    message = service.message
    service._progress((Event(), 'late', 1, 1))
    assert service.message == message
    service.close()


def test_ordinary_global_validation_failure_discards_all_results(qtbot, tmp_path, monkeypatch):
    source, _ = make_package(tmp_path)
    from phy.apps.large_recording.refractory import RefractoryAccumulator
    from phy.cluster.clustering import Clustering

    clustering = Clustering(np.load(source / 'spike_clusters.npy'))
    original = RefractoryAccumulator.finish
    changed = []

    def finish(accumulator, cancel=None):
        value = original(accumulator, cancel)
        if not changed:
            with (source / 'params.py').open('a') as stream:
                stream.write('\n# exposure changed during computation\n')
            changed.append(True)
        return value

    monkeypatch.setattr(RefractoryAccumulator, 'finish', finish)
    service = refractory_service.RefractoryService(
        Bunch(
            dir_path=source,
            supervisor=Bunch(clustering=clustering),
        )
    )
    try:
        service.request([0, 1])
        qtbot.waitUntil(lambda: service.message.startswith('RP unavailable:'), timeout=10000)
        assert changed and not service.results
        assert 'changed' in service.message
    finally:
        worker = service._worker
        service.close()
        qtbot.waitUntil(lambda: not worker.thread.is_alive(), timeout=10000)


def test_ordinary_nonfinite_unit_preserves_healthy_unit(qtbot, tmp_path):
    source, _ = make_package(tmp_path)
    from phy.cluster.clustering import Clustering

    amp = np.load(source / 'amplitudes.npy')
    amp[0] = np.nan
    np.save(source / 'amplitudes.npy', amp)
    service = refractory_service.RefractoryService(
        Bunch(
            dir_path=source,
            supervisor=Bunch(clustering=Clustering(np.load(source / 'spike_clusters.npy'))),
        )
    )
    try:
        service.request([0, 1])
        qtbot.waitUntil(lambda: bool(service.results), timeout=10000)
        assert service.results[0]['status'] == 'unavailable'
        assert service.results[1]['status'] == 'ready'
    finally:
        worker = service._worker
        service.close()
        qtbot.waitUntil(lambda: not worker.thread.is_alive(), timeout=10000)


def test_ordinary_cache_hit_still_rejects_final_source_mutation(qtbot, tmp_path, monkeypatch):
    source, _ = make_package(tmp_path)
    from phy.cluster.clustering import Clustering

    service = refractory_service.RefractoryService(
        Bunch(
            dir_path=source,
            supervisor=Bunch(clustering=Clustering(np.load(source / 'spike_clusters.npy'))),
        )
    )
    try:
        service.request([0])
        qtbot.waitUntil(lambda: bool(service.results), timeout=10000)
        original = refractory_service.hashlib.sha256
        changed = []

        def mutate(*args, **kwargs):
            if not changed:
                with (source / 'params.py').open('a') as stream:
                    stream.write('\n# changed while hashing a cached membership\n')
                changed.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(refractory_service.hashlib, 'sha256', mutate)
        service.request([], enabled=False)
        service.request([0])
        qtbot.waitUntil(lambda: service.message.startswith('RP unavailable:'), timeout=10000)
        assert changed and not service.results
    finally:
        worker = service._worker
        service.close()
        qtbot.waitUntil(lambda: not worker.thread.is_alive(), timeout=10000)
