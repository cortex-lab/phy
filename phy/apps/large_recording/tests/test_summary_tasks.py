"""Store task adapters preserve curation and recover detached artifacts."""

import json

from phy.apps.large_recording.summary_store import (
    estimate_summary_store,
    summary_store_reference,
)
from phy.apps.large_recording.summary_tasks import (
    StoreRequest,
    connect_summary_store,
    inspect_summary_store,
    prepare_summary_store,
)
from phy.apps.large_recording.tests.test_live_summaries import package
from phy.gui.tasks import BackgroundTaskManager, TaskSpec


def execute(qtbot, manager, key, worker, request):
    outcomes = []
    manager.finished.connect(outcomes.append)
    manager.submit(TaskSpec(key, key, worker, request))
    qtbot.waitUntil(lambda: bool(outcomes))
    manager.finished.disconnect(outcomes.append)
    return outcomes[0]


def test_estimate_detached_build_then_attach(qtbot, tmp_path):
    _, bounded = package(tmp_path)
    manager = BackgroundTaskManager()
    request = StoreRequest(str(bounded), destination=str(tmp_path / 'store'))
    inspection = execute(qtbot, manager, 'inspect', inspect_summary_store, request)
    assert inspection.state == 'succeeded'
    assert not inspection.result.compatible
    assert inspection.result.source_available
    estimate = estimate_summary_store(bounded)
    assert estimate == inspection.result.estimate
    built = execute(qtbot, manager, 'build', prepare_summary_store, request)
    assert built.state == 'succeeded'
    assert summary_store_reference(bounded) is None
    index = json.loads((tmp_path / 'store' / 'index.json').read_text())
    assert index['payload_bytes'] == estimate.payload_bytes
    attached = execute(qtbot, manager, 'attach', connect_summary_store, request)
    assert attached.state == 'succeeded'
    assert summary_store_reference(bounded) is not None
    inspection = execute(qtbot, manager, 'inspect-again', inspect_summary_store, request)
    assert inspection.result.compatible


def test_attach_cannot_overwrite_a_newer_choice(qtbot, tmp_path):
    from phy.apps.large_recording.summary_store import attach_summary_store, build_summary_store

    _, bounded = package(tmp_path)
    first = build_summary_store(bounded, tmp_path / 'first', attach=False)
    second = build_summary_store(bounded, tmp_path / 'second', attach=False)
    request = StoreRequest(str(bounded), destination=str(first), expected_reference=None)
    attach_summary_store(bounded, second)
    reference = summary_store_reference(bounded)
    outcome = execute(
        qtbot, BackgroundTaskManager(), 'attach-conflict', connect_summary_store, request
    )
    assert outcome.state == 'failed'
    assert 'changed while' in outcome.error
    assert summary_store_reference(bounded) == reference
    assert first.is_dir()


def test_post_publish_warning_keeps_artifact_available(qtbot, tmp_path, monkeypatch):
    from phy.apps.large_recording import summary_store

    _, bounded = package(tmp_path)
    output = tmp_path / 'store'
    original = summary_store._fsync_directory

    def fail_parent(path):
        if path == output.parent and output.exists():
            raise OSError('parent sync failed')
        original(path)

    monkeypatch.setattr(summary_store, '_fsync_directory', fail_parent)
    outcome = execute(
        qtbot,
        BackgroundTaskManager(),
        'published-warning',
        prepare_summary_store,
        StoreRequest(str(bounded), destination=str(output)),
    )
    assert outcome.state == 'succeeded'
    assert outcome.result.path == str(output)
    assert 'parent sync failed' in outcome.result.warning
    assert output.is_dir()
    assert summary_store_reference(bounded) is None


def test_attachment_lock_wait_is_cancellable(qtbot, tmp_path):
    from phy.apps.large_recording.curation import _commit_lock
    from phy.apps.large_recording.summary_store import build_summary_store

    _, bounded = package(tmp_path)
    store = build_summary_store(bounded, tmp_path / 'store', attach=False)
    lock_path = bounded / '.phy' / 'summary-store-attachment'
    lock_path.mkdir(parents=True, exist_ok=True)
    manager = BackgroundTaskManager()
    outcomes = []
    manager.finished.connect(outcomes.append)
    with _commit_lock(lock_path):
        manager.submit(
            TaskSpec(
                'locked-attach',
                'Attach',
                connect_summary_store,
                StoreRequest(str(bounded), destination=str(store)),
                heavy=False,
            )
        )
        qtbot.wait(100)
        manager.cancel()
        qtbot.waitUntil(lambda: bool(outcomes), timeout=2000)
    assert outcomes[0].state == 'cancelled'
    assert summary_store_reference(bounded) is None


def test_oversized_bad_reference_can_be_explicitly_replaced(qtbot, tmp_path):
    from phy.apps.large_recording.summary_store import build_summary_store

    _, bounded = package(tmp_path)
    store = build_summary_store(bounded, tmp_path / 'store', attach=False)
    path = bounded / '.phy' / 'live-summary-store.json'
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b'x' * 65537)
    snapshot = summary_store_reference(bounded, allow_invalid=True)
    outcome = execute(
        qtbot,
        BackgroundTaskManager(),
        'repair-reference',
        connect_summary_store,
        StoreRequest(str(bounded), destination=str(store), expected_reference=snapshot),
    )
    assert outcome.state == 'succeeded'
    assert len(summary_store_reference(bounded)) < 65536
