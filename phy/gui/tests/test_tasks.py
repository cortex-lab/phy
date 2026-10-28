"""Task publication boundaries, bounded delivery, ETA and cancellation."""

from dataclasses import dataclass
from threading import Event, get_ident

import pytest

from phy.gui.tasks import BackgroundTaskManager, StageETA, TaskProgress, TaskSpec, heavy_io


@dataclass(frozen=True)
class Request:
    value: int


def test_latest_progress_is_bounded_and_delivery_is_on_gui(qtbot):
    owner = get_ident()
    release, reported = Event(), Event()
    manager = BackgroundTaskManager()
    changes, outcomes = [], []
    manager.progress.connect(lambda p: changes.append((get_ident(), p)))
    manager.finished.connect(lambda o: outcomes.append((get_ident(), o)))

    def worker(request, context):
        assert get_ident() != owner
        for n in range(20000):
            context.report(TaskProgress('Counting', n, 20000, 'items'))
        reported.set()
        assert release.wait(5)
        return request.value

    manager.submit(TaskSpec('count', 'Count', worker, Request(7)))
    assert reported.wait(5)  # GUI deliberately stalled while worker reports.
    assert len(manager._mailbox) == 1
    assert not changes
    release.set()
    qtbot.waitUntil(lambda: bool(outcomes))
    assert outcomes[0][0] == owner
    assert outcomes[0][1].result == 7
    assert changes[-1][1].done == 19999
    assert all(thread == owner for thread, _ in changes)


def test_duplicate_task_and_precommit_cancel(qtbot):
    manager, other = BackgroundTaskManager(), BackgroundTaskManager()
    started, release = Event(), Event()
    outcomes = []
    manager.finished.connect(outcomes.append)

    def worker(request, context):
        started.set()
        assert release.wait(5)
        context.check_cancel()
        return None

    spec = TaskSpec('duplicate', 'Duplicate', worker, Request(1))
    manager.submit(spec)
    assert started.wait(5)
    with pytest.raises(ValueError, match='already'):
        other.submit(spec)
    manager.cancel()
    release.set()
    qtbot.waitUntil(lambda: bool(outcomes))
    assert outcomes[0].state == 'cancelled'


def test_cancel_after_commit_preserves_success_and_detach_suppresses_callbacks(qtbot):
    committed, release = Event(), Event()
    manager = BackgroundTaskManager()
    outcomes = []
    manager.finished.connect(outcomes.append)

    def worker(request, context):
        context.commit()
        committed.set()
        assert release.wait(5)
        context.check_cancel()  # A published artifact cannot become cancellation.
        return 'published'

    manager.submit(TaskSpec('commit', 'Commit', worker, Request(1)))
    assert committed.wait(5)
    manager.cancel()
    release.set()
    qtbot.waitUntil(lambda: bool(outcomes))
    assert outcomes[0].state == 'succeeded'
    assert outcomes[0].result == 'published'
    # Detached owners receive no completion and never block shutdown.
    committed.clear()
    release.clear()
    outcomes.clear()
    manager.submit(TaskSpec('detach', 'Detach', worker, Request(1)))
    assert committed.wait(5)
    manager.detach()
    release.set()
    qtbot.waitUntil(lambda: not manager.busy)
    assert not outcomes


def test_heavy_io_is_shared_and_waiting_task_can_cancel(qtbot):
    manager = BackgroundTaskManager()
    outcomes = []
    manager.finished.connect(outcomes.append)
    entered = Event()

    def worker(request, context):
        entered.set()

    with heavy_io(lambda: None):
        manager.submit(TaskSpec('io', 'IO', worker, Request(1)))
        manager.cancel()
        qtbot.waitUntil(lambda: bool(outcomes))
    assert outcomes[0].state == 'cancelled'
    assert not entered.is_set()


def test_errors_and_mutable_requests(qtbot):
    manager = BackgroundTaskManager()
    with pytest.raises(TypeError, match='immutable'):
        manager.submit(TaskSpec('mutable', 'Mutable', lambda r, c: None, {}))
    outcomes = []
    manager.finished.connect(outcomes.append)

    def worker(request, context):
        raise OSError('disk full')

    manager.submit(TaskSpec('error', 'Error', worker, Request(1)))
    qtbot.waitUntil(lambda: bool(outcomes))
    assert outcomes[0].state == 'failed'
    assert outcomes[0].error == 'disk full'


def test_eta_resets_per_stage_and_uses_injected_clock():
    now = [0.0]
    eta = StageETA(clock=lambda: now[0])
    assert eta.update(TaskProgress('Validate', 0, 100, 'bytes')) is None
    now[0] = 1
    assert eta.update(TaskProgress('Validate', 10, 100, 'bytes')) is None
    now[0] = 3
    assert eta.update(TaskProgress('Validate', 30, 100, 'bytes')) == 7
    now[0] = 4
    assert eta.update(TaskProgress('Build', 5, 100, 'rows')) is None
    now[0] = 7
    assert eta.update(TaskProgress('Build', 20, 100, 'rows')) == 16
    assert eta.update(TaskProgress('Finalize')) is None
