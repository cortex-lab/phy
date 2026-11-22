"""Stable transaction capture, worker acknowledgement, and explicit draft lifecycle."""

import json
import threading
from concurrent.futures import CancelledError

import numpy as np
import pytest
from phylib.utils import connect, unconnect

from phy.apps.large_recording import recovery, recovery_gui
from phy.apps.large_recording.curation import BoundedSession
from phy.apps.large_recording.tests.test_adapter import make_controller
from phy.gui.qt import mock_dialogs


def protected(qtbot, coordinator):
    qtbot.waitUntil(
        lambda: (
            coordinator.status['protected_revision'] == coordinator.status['current_revision']
            and coordinator.status['state'] == 'protected'
        ),
        timeout=5000,
    )


def latest(coordinator):
    folder = next(iter(coordinator.root.glob('*'))) / coordinator.session_id
    heads = [json.loads(path.read_text())['payload'] for path in folder.glob('head-*.json')]
    return max(heads, key=lambda head: head['revision'])


def test_capture_has_no_spike_work_and_waits_for_outer_transaction(qtbot, tmp_path, monkeypatch):
    _, controller = make_controller(tmp_path)
    coordinator = controller.recovery
    coordinator.root = tmp_path / 'recovery'
    coordinator.start()
    supervisor = controller.supervisor
    writes, owner = [], threading.get_ident()
    original = recovery.RecoveryStore.write

    def write(store, payload):
        writes.append(threading.get_ident())
        return original(store, payload)

    def forbid(*args, **kwargs):
        pytest.fail('Recovery capture scanned or copied spike state.')

    try:
        monkeypatch.setattr(recovery.RecoveryStore, 'write', write)
        monkeypatch.setattr(controller.bounded_session, 'snapshot', forbid)
        monkeypatch.setattr(recovery, 'project_assignments', forbid)
        supervisor.label('group', 'mua', [0])
        future = coordinator.request_flush()
        qtbot.waitUntil(future.done)
        protected(qtbot, coordinator)
        assert writes and all(thread != owner for thread in writes)
        assert latest(coordinator)['events'][-1]['value'] == 'mua'
        assert not (controller.dir_path / '.phy-curation' / 'current.json').exists()
    finally:
        controller.close()


def test_failed_observer_never_protects_provisional_merge(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    coordinator = controller.recovery
    coordinator.root = tmp_path / 'recovery'
    coordinator.start()
    supervisor = controller.supervisor

    @connect(event='bounded_state_changed', sender=supervisor)
    def reject(sender):
        assert coordinator.request_flush() is None
        raise RuntimeError('observer rolls back')

    try:
        with pytest.raises(RuntimeError, match='rolls back'):
            supervisor.merge([0, 1], to=100)
        unconnect(reject)
        future = coordinator.request_flush()
        qtbot.waitUntil(future.done)
        protected(qtbot, coordinator)
        draft = latest(coordinator)
        assert draft['events'] == []
        assert draft['next_cluster_id'] >= 101
        assert supervisor.session.events == []
    finally:
        unconnect(reject)
        controller.close()


def test_one_active_one_pending_newest_snapshot(qtbot, tmp_path, monkeypatch):
    _, controller = make_controller(tmp_path)
    coordinator = controller.recovery
    coordinator.root = tmp_path / 'recovery'
    coordinator.start()
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = recovery.RecoveryStore.write

    def slow(store, payload):
        calls.append(payload['revision'])
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return original(store, payload)

    monkeypatch.setattr(recovery.RecoveryStore, 'write', slow)
    try:
        supervisor = controller.supervisor
        supervisor.label('group', 'mua', [0])
        first = coordinator.request_flush()
        qtbot.waitUntil(entered.is_set)
        supervisor.label('group', 'noise', [0])
        middle = coordinator.request_flush()
        supervisor.label('group', 'good', [0])
        last = coordinator.request_flush()
        assert middle.cancelled()
        assert coordinator.status['protected_revision'] == 0
        release.set()
        qtbot.waitUntil(last.done)
        protected(qtbot, coordinator)
        assert first.done() and calls == [1, 3]
        assert latest(coordinator)['events'][-1]['value'] == 'good'
    finally:
        release.set()
        controller.close()


def test_failed_fsync_never_acknowledges_protection_and_retry_succeeds(
    qtbot, tmp_path, monkeypatch
):
    _, controller = make_controller(tmp_path)
    coordinator = controller.recovery
    coordinator.root = tmp_path / 'recovery'
    coordinator.start()
    controller.supervisor.label('group', 'mua', [0])
    original = recovery._atomic_json

    def fail(*args, **kwargs):
        raise OSError('disk full')

    try:
        monkeypatch.setattr(recovery, '_atomic_json', fail)
        failed = coordinator.request_flush()
        qtbot.waitUntil(failed.done)
        qtbot.waitUntil(lambda: coordinator.status['state'] == 'error')
        assert coordinator.status['protected_revision'] == 0
        assert coordinator.status['last_protected_at'] is None
        assert 'disk full' in coordinator.status['error']
        assert controller.supervisor.is_dirty()
        monkeypatch.setattr(recovery, '_atomic_json', original)
        succeeded = coordinator.request_flush()
        qtbot.waitUntil(succeeded.done)
        protected(qtbot, coordinator)
    finally:
        controller.close()


def test_explicit_save_with_newer_edit_keeps_draft_and_dirty(qtbot, tmp_path, monkeypatch):
    from phy.apps.large_recording import curation

    _, controller = make_controller(tmp_path)
    coordinator = controller.recovery
    coordinator.root = tmp_path / 'recovery'
    coordinator.start()
    supervisor = controller.supervisor
    supervisor.label('group', 'mua', [0])
    original = curation._write_json
    once = []

    def edit_during_save(path, value):
        original(path, value)
        if path.name == 'journal.json' and not once:
            once.append(True)
            supervisor.label('notes', 'newer than saved checkpoint', [0])

    try:
        monkeypatch.setattr(curation, '_write_json', edit_during_save)
        supervisor.save()
        assert supervisor.is_dirty()
        assert len(supervisor.session.events) == 2
        with BoundedSession(controller.dir_path) as saved:
            assert len(saved.events) == 1
        future = coordinator.request_flush()
        qtbot.waitUntil(future.done)
        protected(qtbot, coordinator)
        draft = latest(coordinator)
        assert len(draft['events']) == 2
        assert draft['base_pointer'] == supervisor.session._expected_pointer
    finally:
        controller.close()


def test_reject_reset_and_undo_protect_review_only_changes(qtbot, tmp_path):
    mapping = {'format_version': '2', 'unit_ids': [0, 1, 2, 99], 'merges': [{'unit_ids': [0, 1]}]}
    _, controller = make_controller(tmp_path, propositions=mapping)
    gui = controller.create_gui(default_views=[], do_prompt_save=False)
    supervisor, coordinator = controller.supervisor, controller.recovery
    key = supervisor.merge_propositions.catalog.propositions[0].key
    try:
        supervisor._reject_merge_proposition(supervisor.merge_propositions_view, key)
        future = coordinator.request_flush()
        qtbot.waitUntil(future.done)
        protected(qtbot, coordinator)
        draft = latest(coordinator)
        assert draft['events'] == []
        assert draft['reviews']['reviews'][key]['decision'] == 'rejected'
        recovered = recovery.build_recovered_state(controller.bounded_session, draft)
        assert (
            recovered.extra_documents['curation_review.json']['reviews'][key]['decision']
            == 'rejected'
        )
        supervisor.undo()
        future = coordinator.request_flush()
        qtbot.waitUntil(future.done)
        protected(qtbot, coordinator)
        assert latest(coordinator)['reviews']['reviews'] == {}
    finally:
        gui.close()
        controller.close()


def test_startup_restore_stays_unsaved_and_discard_is_durable(qtbot, tmp_path, monkeypatch):
    _, controller = make_controller(tmp_path)
    path = controller.dir_path
    root = tmp_path / 'recovery'
    coordinator = controller.recovery
    coordinator.root = root
    coordinator.start()
    controller.supervisor.merge([0, 1], to=100)
    controller.supervisor.label('group', 'mua', [100])
    future = coordinator.request_flush()
    qtbot.waitUntil(future.done)
    protected(qtbot, coordinator)
    controller.close()
    qtbot.waitUntil(lambda: not coordinator._writer.thread.is_alive())
    from phy.apps.template.gui import TemplateController

    reopened = TemplateController(
        dir_path=path, config_dir=tmp_path / 'new-config', recovery_root=root, plugins=[]
    )
    gui = None
    try:
        with mock_dialogs('restore'):
            gui = reopened.create_gui(default_views=[], do_prompt_save=False, offer_recovery=True)
        assert reopened.supervisor.is_dirty()
        assert reopened.supervisor._global_history.current_position == 0
        assert reopened.bounded_session.metadata['group'][100] == 'mua'
        assert reopened.bounded_session.next_cluster_id == 101
        assert not (path / '.phy-curation' / 'current.json').exists()
        future = reopened.recovery.discard()
        qtbot.waitUntil(future.done)
        future.result()
        reader = recovery.RecoveryStore(reopened.bounded_session, root)
        try:
            assert not reader.candidates(None)[0]
        finally:
            reader.close()
    finally:
        if gui is not None:
            gui.close()
        reopened.close()


def test_ready_disabled_save_and_dirty_clean_notifications(qtbot, tmp_path):
    _, controller = make_controller(tmp_path)
    coordinator, supervisor = controller.recovery, controller.supervisor
    assert coordinator.status['state'] == 'ready'
    assert coordinator.status['protected_revision'] == 0
    assert coordinator.status['last_protected_at'] is None
    dirty = []
    coordinator.changed.connect(lambda: dirty.append(supervisor.is_dirty()))
    try:
        supervisor.label('group', 'mua', [0])
        assert supervisor.is_dirty()
        supervisor.undo()
        assert not supervisor.is_dirty()
        supervisor.label('group', 'noise', [0])
        assert supervisor.is_dirty()
        assert dirty == [True, False, True]
        path = controller.dir_path
    finally:
        controller.close()
    from phy.apps.template.gui import TemplateController

    disabled = TemplateController(
        dir_path=path, config_dir=tmp_path / 'disabled', recovery_enabled=False, plugins=[]
    )
    try:
        disabled.supervisor.label('group', 'mua', [0])
        disabled.supervisor.save()
        assert disabled.recovery._writer is None
        assert disabled.recovery.status['state'] == 'disabled'
        assert disabled.recovery.status['last_checkpoint_at'] is not None
    finally:
        disabled.close()


def test_stale_completions_cannot_replace_newer_protection_or_error(qtbot, tmp_path):
    from concurrent.futures import Future

    _, controller = make_controller(tmp_path)
    coordinator = controller.recovery
    coordinator.start()
    try:
        for group in ('mua', 'noise'):
            controller.supervisor.label('group', group, [0])
            future = coordinator.request_flush()
            qtbot.waitUntil(future.done)
            protected(qtbot, coordinator)
        before = dict(coordinator.status)
        old_error = Future()
        old_error.set_exception(OSError('old failed write'))
        coordinator._completed(('snapshot', 1, old_error))
        assert coordinator.status == before
        controller.supervisor.label('group', 'good', [0])
        new_error = Future()
        new_error.set_exception(OSError('new failed write'))
        coordinator._completed(('snapshot', 3, new_error))
        stale_success = Future()
        stale_success.set_result({'retired': False, 'protected_at': 1, 'base_pointer': None})
        coordinator._completed(('snapshot', 1, stale_success))
        assert coordinator.status['state'] == 'error'
        assert coordinator.status['error'] == 'new failed write'
        assert coordinator.status['protected_revision'] == 2
    finally:
        controller.close()


def test_save_rebases_only_own_newer_queued_snapshot(qtbot, tmp_path, monkeypatch):
    _, controller = make_controller(tmp_path)
    coordinator, supervisor = controller.recovery, controller.supervisor
    coordinator.start()
    entered, release = threading.Event(), threading.Event()
    original = recovery.RecoveryStore.write
    calls = []

    def slow(store, payload):
        calls.append(payload['revision'])
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return original(store, payload)

    monkeypatch.setattr(recovery.RecoveryStore, 'write', slow)
    try:
        supervisor.label('group', 'mua', [0])
        coordinator.request_flush()
        qtbot.waitUntil(entered.is_set)
        supervisor.save()
        supervisor.label('group', 'noise', [0])
        future = coordinator.request_flush()
        release.set()
        qtbot.waitUntil(future.done)
        protected(qtbot, coordinator)
        assert latest(coordinator)['base_pointer'] == supervisor.session._expected_pointer
        assert latest(coordinator)['events'][-1]['value'] == 'noise'
        assert supervisor.is_dirty()
        assert coordinator.status['warning'] is None
    finally:
        release.set()
        controller.close()


@pytest.mark.parametrize('choice', ['save', 'close', 'cancel'])
def test_actual_close_save_discard_cancel_contract(qtbot, tmp_path, choice):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=[], do_prompt_save=True, offer_recovery=False)
    coordinator, supervisor = controller.recovery, controller.supervisor
    try:
        supervisor.label('group', 'mua', [0])
        future = coordinator.request_flush()
        qtbot.waitUntil(future.done)
        protected(qtbot, coordinator)
        with mock_dialogs(choice):
            gui.close()
        if choice == 'cancel':
            assert not gui._closed
            assert coordinator._started
            assert coordinator._writer.thread.is_alive()
            assert supervisor.is_dirty()
        else:
            assert gui._closed
            assert not coordinator._started
            pointer = controller.dir_path / '.phy-curation' / 'current.json'
            assert pointer.exists() == (choice == 'save')
            assert supervisor.is_dirty() == (choice == 'close')
    finally:
        if not gui._closed:
            with mock_dialogs('close'):
                gui.close()
        controller.close()


def test_failed_discard_keeps_gui_and_worker_alive(qtbot, tmp_path, monkeypatch):
    _, controller = make_controller(tmp_path)
    gui = controller.create_gui(default_views=[], do_prompt_save=True, offer_recovery=False)
    coordinator = controller.recovery
    original = recovery.RecoveryStore.retire
    try:
        controller.supervisor.label('group', 'mua', [0])
        future = coordinator.request_flush()
        qtbot.waitUntil(future.done)
        protected(qtbot, coordinator)

        def fail(*args, **kwargs):
            raise OSError('discard is not durable')

        monkeypatch.setattr(recovery.RecoveryStore, 'retire', fail)
        with mock_dialogs('close'):
            gui.close()
        assert not gui._closed
        assert coordinator._started
        qtbot.waitUntil(lambda: coordinator.status['state'] == 'error')
        assert 'discard is not durable' in coordinator.status['error']
    finally:
        monkeypatch.setattr(recovery.RecoveryStore, 'retire', original)
        if not gui._closed:
            with mock_dialogs('close'):
                gui.close()
        controller.close()


def test_post_pointer_fsync_failure_reports_uncertain_save(qtbot, tmp_path, monkeypatch):
    from phy.apps.large_recording import curation

    _, controller = make_controller(tmp_path)
    coordinator, supervisor = controller.recovery, controller.supervisor
    original = curation._fsync_directory
    root = controller.dir_path / '.phy-curation'
    once = []

    def fail_after_pointer(path):
        if path == root and (root / 'current.json').exists() and not once:
            once.append(True)
            raise OSError('directory sync failed after pointer publication')
        return original(path)

    try:
        supervisor.label('group', 'mua', [0])
        monkeypatch.setattr(curation, '_fsync_directory', fail_after_pointer)
        with pytest.raises(OSError, match='directory sync failed'):
            supervisor.save()
        assert supervisor.is_dirty()
        assert coordinator.status['state'] == 'error'
        assert coordinator.status['last_checkpoint_at'] is None
        assert 'durability is uncertain' in coordinator.status['error']
        with BoundedSession(controller.dir_path) as saved:
            assert saved.metadata['group'][0] == 'mua'
        assert supervisor.session._expected_pointer is None
    finally:
        controller.close()
