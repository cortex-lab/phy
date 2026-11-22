"""Private recovery heads survive interruption without publishing curation."""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from phy.apps.large_recording import recovery
from phy.apps.large_recording.curation import BoundedSession, _digest
from phy.apps.large_recording.recovery import (
    FORMAT,
    VERSION,
    RecoveryStore,
    build_recovered_state,
)
from phy.apps.large_recording.tests.test_curation import make_session


def payload(session, store, revision):
    return {
        'format': FORMAT,
        'version': VERSION,
        'session_id': store.session_id,
        'revision': revision,
        'created_at': time.time(),
        'bindings': store.bindings,
        'base_pointer': session._expected_pointer,
        'events': list(session.events),
        'next_cluster_id': session.next_cluster_id,
        'reviews': None,
    }


def test_two_heads_fallback_retirement_and_active_owner(tmp_path):
    source, session = make_session(tmp_path)
    root = tmp_path / 'recovery'
    first = RecoveryStore(session, root)
    reader = RecoveryStore(session, root)
    try:
        session.record_label('group', [0], 'mua')
        first.write(payload(session, first, 1))
        session.record_label('group', [0], 'good')
        first.write(payload(session, first, 2))
        assert not reader.candidates(None)[0]  # Live owner is never offered.
        first.close()
        newest = max(
            first.branch.glob('head-*.json'),
            key=lambda p: json.loads(p.read_text())['payload']['revision'],
        )
        newest.write_text('{partial')
        drafts, warnings = reader.candidates(None)
        assert warnings and len(drafts) == 1
        assert drafts[0]['payload']['revision'] == 1
        recovered = build_recovered_state(session, drafts[0]['payload'])
        assert recovered.metadata['group'][0] == 'mua'
        assert session.metadata['group'][0] == 'good'
        reader.retire(1, branch=first.branch)
        assert not reader.candidates(None)[0]
    finally:
        first.close()
        reader.close()
        session.close()
        source.close()


@pytest.mark.parametrize('phase', ['before_replace', 'after_replace'])
def test_process_exit_during_snapshot_keeps_a_verified_head(tmp_path, phase):
    source, session = make_session(tmp_path)
    store = RecoveryStore(session, tmp_path / 'recovery')
    session.record_label('group', [0], 'mua')
    store.write(payload(session, store, 1))
    session.record_label('group', [0], 'good')
    item = payload(session, store, 2)
    item_path = tmp_path / 'payload.json'
    item_path.write_text(json.dumps(item))
    store.close()
    script = """
import json, os, sys
from phy.apps.large_recording import recovery
from phy.apps.large_recording.curation import BoundedSession
session = BoundedSession(sys.argv[1])
item = json.load(open(sys.argv[3]))
store = recovery.RecoveryStore(session, sys.argv[2], session_id=item['session_id'])
if sys.argv[4] == 'before_replace':
    original = recovery._write_json
    def fail(path, value):
        original(path, value)
        os._exit(91)
    recovery._write_json = fail
else:
    recovery._fsync_directory = lambda path: os._exit(92)
store.write(item)
"""
    try:
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                script,
                str(session.path),
                str(tmp_path / 'recovery'),
                str(item_path),
                phase,
            ],
            timeout=30,
            capture_output=True,
        )
        assert result.returncode == (91 if phase == 'before_replace' else 92)
        reader = RecoveryStore(session, tmp_path / 'recovery')
        try:
            drafts, _ = reader.candidates(None)
            assert len(drafts) == 1
            head = drafts[0]['payload']
            assert head['revision'] == (1 if phase == 'before_replace' else 2)
            expected = 'mua' if phase == 'before_replace' else 'good'
            assert build_recovered_state(session, head).metadata['group'][0] == expected
        finally:
            reader.close()
    finally:
        session.close()
        source.close()


def test_full_active_journal_recovers_undo_branch_and_allocator(tmp_path):
    source, session = make_session(tmp_path)
    store = RecoveryStore(session, tmp_path / 'recovery')
    try:
        baseline = session.journal_state()
        session.record_merge([0, 1], 100)
        session.save()
        session.restore_journal(baseline)
        session.record_merge([1, 2], 101)
        item = payload(session, store, 1)
        recovered = build_recovered_state(session, item)
        assert recovered.events == session.events
        assert recovered.next_cluster_id == 102
        np.testing.assert_array_equal(recovered.assignments, session.assignments)
        with BoundedSession(session.path) as saved:
            assert saved.events[0]['target'] == 100
        item['next_cluster_id'] = 101
        with pytest.raises(ValueError, match='allocator'):
            build_recovered_state(session, item)
        item['next_cluster_id'] = 102
        item['bindings'] = dict(item['bindings'], source_revision='changed')
        with pytest.raises(ValueError, match='bindings'):
            build_recovered_state(session, item)
    finally:
        store.close()
        session.close()
        source.close()


def test_recovery_is_not_seen_by_session_load_or_apply(tmp_path):
    from phy.apps.large_recording.replay import apply_curation

    source, session = make_session(tmp_path)
    store = RecoveryStore(session, tmp_path / 'recovery')
    try:
        session.record_label('group', [0], 'mua')
        session.save()
        session.record_label('group', [0], 'noise')
        store.write(payload(session, store, 1))
        with BoundedSession(session.path) as saved:
            assert saved.metadata['group'][0] == 'mua'
        output = apply_curation(
            source.params_path, session.path, tmp_path / 'output', batch_size=5
        )
        rows = (output / 'cluster_group.tsv').read_text()
        assert '0\tmua' in rows and '0\tnoise' not in rows
    finally:
        store.close()
        session.close()
        source.close()


def test_other_checkpoint_never_rebases_or_deletes_a_draft(tmp_path):
    source, session = make_session(tmp_path)
    store = RecoveryStore(session, tmp_path / 'recovery')
    session.record_label('group', [0], 'mua')
    store.write(payload(session, store, 1))
    store.close()
    try:
        session.save()
        reader = RecoveryStore(session, tmp_path / 'recovery')
        try:
            drafts, warnings = reader.candidates(session._expected_pointer)
            assert not drafts and warnings
            assert list(store.branch.glob('head-*.json'))
        finally:
            reader.close()
    finally:
        session.close()
        source.close()


def test_skipped_revisions_preserve_two_independent_heads(tmp_path):
    source, session = make_session(tmp_path)
    store = RecoveryStore(session, tmp_path / 'recovery')
    try:
        session.record_label('group', [0], 'mua')
        store.write(payload(session, store, 1))
        session.record_label('group', [0], 'good')
        store.write(payload(session, store, 3))
        heads = list(store.branch.glob('head-*.json'))
        assert len(heads) == 2
        assert {json.loads(path.read_text())['payload']['revision'] for path in heads} == {1, 3}
        newest = max(heads, key=lambda p: json.loads(p.read_text())['payload']['revision'])
        newest.write_text('{torn')
        store.close()
        reader = RecoveryStore(session, tmp_path / 'recovery')
        try:
            drafts, _ = reader.candidates(None)
            assert drafts[0]['payload']['revision'] == 1
        finally:
            reader.close()
    finally:
        store.close()
        session.close()
        source.close()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX mode bits')
def test_private_permissions_and_new_ancestor_directory_sync(tmp_path, monkeypatch):
    source, session = make_session(tmp_path)
    root = tmp_path / 'new-parent' / 'app' / 'recovery'
    synced = []
    original = recovery._fsync_directory

    def sync(path):
        synced.append(path)
        original(path)

    monkeypatch.setattr(recovery, '_fsync_directory', sync)
    store = RecoveryStore(session, root)
    try:
        store.write(payload(session, store, 1))
        store.retire(0)
        for path in (root, store.directory, store.branch):
            assert path.stat().st_mode & 0o777 == 0o700
        for name in ('owner.lock', 'head-0.json', 'retired.json'):
            assert (store.branch / name).stat().st_mode & 0o777 == 0o600
        assert tmp_path in synced
        assert root.parent in synced
    finally:
        store.close()
        session.close()
        source.close()
