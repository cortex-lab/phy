"""Private semantic recovery drafts; never a source for committed curation replay."""

import copy
import json
import logging
import math
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from ._immutable import freeze
from .curation import (
    BoundedSession,
    _current_pointer,
    _digest,
    _fsync_directory,
    _integer,
    _read_json,
    _safe_file,
    _write_json,
    interpret_events,
    normalize_metadata,
    project_assignments,
)

logger = logging.getLogger(__name__)
FORMAT = 'phy_bounded_recovery'
VERSION = 1
SOFT_BUDGET = 256 * 1024 * 1024


def default_recovery_root():
    override = os.environ.get('PHY_RECOVERY_ROOT')
    if override:
        return Path(override).expanduser()
    if sys.platform == 'win32':
        root = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    elif sys.platform == 'darwin':
        root = Path.home() / 'Library' / 'Application Support'
    else:
        root = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
    return root / 'phy' / 'recovery'


def recovery_bindings(session):
    return freeze(
        {
            'manifest_sha256': _digest(session.manifest),
            'source_identity': session.manifest['source']['identity']['sha256'],
            'source_revision': session.manifest['source']['revision'],
            'proposition_documents': session.proposition_documents,
        }
    )


class OwnerLock:
    """A live process owns a branch; PID files and timestamps are not locks."""

    def __init__(self, path):
        flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
        descriptor = os.open(path, flags, 0o600)
        self.stream = os.fdopen(descriptor, 'r+b')
        if os.name != 'nt':
            os.fchmod(descriptor, 0o600)
        try:
            if os.name == 'nt':
                import msvcrt

                self.stream.seek(0, os.SEEK_END)
                if not self.stream.tell():
                    self.stream.write(b'\0')
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            raise

    def close(self):
        if self.stream.closed:
            return
        if os.name == 'nt':
            import msvcrt

            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def _private_directory(path):
    """Create private ancestors and persist every newly created directory entry."""
    path = Path(path)
    missing, parent = [], path
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Recovery storage must be a real directory.')
    if os.name != 'nt' and path.stat().st_mode & 0o077:
        # A supplied root may be a shared directory such as /tmp. Never chmod
        # caller-owned ancestors or turn them into private application storage.
        raise ValueError('Recovery storage must be a private directory (0700).')


def _atomic_json(directory, name, value):
    target = _safe_file(directory, name)
    temporary = _safe_file(directory, f'.{uuid.uuid4().hex}.tmp')
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        _write_json(temporary, value)
        os.replace(temporary, target)
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _retired_through(branch):
    try:
        record = _read_json(_safe_file(branch, 'retired.json'))
    except FileNotFoundError:
        return -1
    if record.get('format') != FORMAT or record.get('version') != VERSION:
        raise ValueError('Recovery retirement record is invalid.')
    return _integer(record['through_revision'], 'Retired recovery revision')


def _verified_heads(branch, bindings, with_slots=False, include_retired=False):
    retired = _retired_through(branch)
    if include_retired:
        retired = -1
    valid, errors = [], []
    for name in ('head-0.json', 'head-1.json'):
        try:
            wrapper = _read_json(_safe_file(branch, name))
            payload = wrapper['payload']
            if wrapper['sha256'] != _digest(payload):
                raise ValueError('Recovery draft checksum mismatch.')
            if payload.get('format') != FORMAT or payload.get('version') != VERSION:
                raise ValueError('Unsupported recovery draft format.')
            if payload['bindings'] != bindings or payload['session_id'] != branch.name:
                raise ValueError('Recovery draft belongs to another preparation/session.')
            revision = _integer(payload['revision'], 'Recovery revision')
            created = payload['created_at']
            if (
                type(created) not in (int, float)
                or not math.isfinite(created)
                or not isinstance(payload['events'], list)
                or payload.get('reviews') is not None
                and not isinstance(payload['reviews'], dict)
            ):
                raise ValueError('Recovery draft header is malformed.')
            datetime.fromtimestamp(created).astimezone()  # Validate displayable timestamps here.
            _integer(payload['next_cluster_id'], 'Recovery allocator')
            if revision > retired:
                valid.append((payload, name) if with_slots else payload)
        except FileNotFoundError:
            continue
        except (ValueError, TypeError, KeyError, OSError, OverflowError, RecursionError) as error:
            errors.append(str(error))
    return sorted(
        valid, key=lambda item: (item[0] if with_slots else item)['revision'], reverse=True
    ), errors


class RecoveryStore:
    """Two independent atomic heads per private writer, outside disposable caches."""

    def __init__(self, session, root=None, session_id=None, soft_budget=SOFT_BUDGET):
        self.bindings = recovery_bindings(session)
        self.dataset_path = Path(session.path)
        root = Path(root or default_recovery_root())
        _private_directory(root)
        self.directory = root / self.bindings['manifest_sha256']
        _private_directory(self.directory)
        self.session_id = session_id or uuid.uuid4().hex
        self.branch = _safe_file(self.directory, self.session_id)
        _private_directory(self.branch)
        self.owner = OwnerLock(_safe_file(self.branch, 'owner.lock'))
        self.soft_budget = soft_budget
        try:
            _fsync_directory(self.directory)
            _fsync_directory(self.directory.parent)
        except BaseException:
            self.owner.close()
            raise

    def write(self, payload):
        if payload['bindings'] != self.bindings or payload['session_id'] != self.session_id:
            raise ValueError('Recovery snapshot bindings changed.')
        revision = _integer(payload['revision'], 'Recovery revision')
        if revision <= _retired_through(self.branch):
            return {'retired': True}
        heads, _ = _verified_heads(self.branch, self.bindings, with_slots=True)
        # Coalescing skips revision numbers: parity would overwrite the only
        # previous head for revisions 1,3. Rotate by the latest verified slot.
        slot = 1 if heads and heads[0][1] == 'head-0.json' else 0
        _atomic_json(
            self.branch,
            f'head-{slot}.json',
            {'sha256': _digest(payload), 'payload': payload},
        )
        # This is a soft budget: never delete the only unresolved draft to meet it.
        size = sum(
            path.stat().st_size
            for branch in self.directory.iterdir()
            if branch.is_dir() and not branch.is_symlink()
            for path in branch.glob('head-*.json')
            if path.is_file() and not path.is_symlink()
        )
        warning = None
        if size > self.soft_budget:
            warning = 'Recovery storage exceeds its soft budget; unresolved drafts were retained.'
        try:
            current = _current_pointer(self.dataset_path / '.phy-curation')
            conflict = current != payload['base_pointer']
        except (OSError, ValueError):
            conflict = True
        return {'retired': False, 'warning': warning, 'checkpoint_conflict': conflict}

    def probe(self):
        _atomic_json(self.branch, 'health.json', {'format': FORMAT, 'checked_at': time.time()})

    def retire(self, through_revision, reason='discarded', branch=None):
        branch = Path(branch or self.branch)
        if branch.parent != self.directory or branch.is_symlink():
            raise ValueError('Recovery branch must belong to this preparation.')
        owner = None
        if branch != self.branch:
            owner = OwnerLock(_safe_file(branch, 'owner.lock'))
        try:
            previous = _retired_through(branch)
            through = max(previous, _integer(through_revision, 'Recovery revision'))
            _atomic_json(
                branch,
                'retired.json',
                {
                    'format': FORMAT,
                    'version': VERSION,
                    'through_revision': through,
                    'reason': reason,
                    'retired_at': time.time(),
                },
            )
            # Retirement is already durable. Cleanup is deliberately narrower:
            # never remove newer, corrupt, unbound, or unknown head contents.
            warning = None
            try:
                self._prune_retired(branch, through)
            except OSError as error:
                warning = f'Recovery draft retired, but storage cleanup failed: {error}'
                logger.warning(warning)
            return {'warning': warning}
        finally:
            if owner is not None:
                owner.close()

    def _prune_retired(self, branch, through):
        heads, _ = _verified_heads(branch, self.bindings, with_slots=True, include_retired=True)
        removed = False
        for payload, name in heads:
            if payload['revision'] <= through:
                _safe_file(branch, name).unlink()
                removed = True
        if removed:
            _fsync_directory(branch)

    def candidates(self, base_pointer):
        candidates, warnings = [], []
        for branch in self.directory.iterdir():
            if not branch.is_dir() or branch.is_symlink() or branch == self.branch:
                continue
            try:
                owner = OwnerLock(_safe_file(branch, 'owner.lock'))
            except OSError:
                continue  # Another live process owns this branch.
            try:
                heads, errors = _verified_heads(branch, self.bindings)
                if heads:
                    latest = heads[0]
                    if latest['base_pointer'] == base_pointer:
                        candidates.append({'branch': str(branch), 'payload': latest})
                    else:
                        warnings.append(
                            'An older-checkpoint recovery branch was retained separately.'
                        )
                warnings.extend(errors)
            except (OSError, ValueError, TypeError, KeyError) as error:
                warnings.append(str(error))
            finally:
                owner.close()
        candidates.sort(key=lambda item: item['payload']['created_at'], reverse=True)
        return candidates, warnings

    def close(self):
        self.owner.close()


def build_recovered_state(session, payload):
    """Reconstruct off the GUI thread without changing live/canonical state."""
    from phy.cluster._propositions import decode_review_mapping

    if payload['bindings'] != recovery_bindings(session):
        raise ValueError('Recovery draft preparation/source bindings do not match.')
    if payload['base_pointer'] != session._expected_pointer:
        raise ValueError('Recovery base checkpoint changed; automatic rebasing is unavailable.')
    session.validate_proposition_documents()
    events = copy.deepcopy(payload['events'])
    if not isinstance(events, list):
        raise ValueError('Recovery journal must be an ordered list.')
    for event in events:
        if event.get('operation') in ('threshold_split', 'restore_parents'):
            event['metadata'] = normalize_metadata(event['metadata'])
    session._validate_event_capabilities(events)
    _, metadata, minimum = interpret_events(session.unit_ids, session.base_metadata, events)
    next_id = _integer(payload['next_cluster_id'], 'Recovery allocator')
    if next_id < max(minimum, session.base_next_cluster_id):
        raise ValueError('Recovery allocator collides with semantic journal IDs.')
    documents = dict(session.extra_documents)
    reviews = payload.get('reviews')
    if reviews is not None:
        source, _ = decode_review_mapping(reviews)
        expected = session.proposition_documents.get('curation.json', {}).get('sha256')
        if (
            source.get('filename') != 'curation.json'
            or source.get('sha256') != expected
            or expected is None
        ):
            raise ValueError('Recovery SLAY ledger is not bound to the producer document.')
        documents['curation_review.json'] = reviews
    elif 'curation.json' in session.proposition_documents:
        raise ValueError('Recovery draft omitted its SLAY ledger.')
    # The detached shell shares only read-only source readers. It owns no file
    # lifetime and must not be closed: adoption keeps the original live session.
    recovered = object.__new__(BoundedSession)
    recovered.__dict__ = dict(session.__dict__)
    recovered.assignments = project_assignments(
        session.base_assignments, events, session.attributes
    )
    recovered.metadata = metadata
    recovered._events = freeze(events)
    recovered.next_cluster_id = max(next_id, session.next_cluster_id)
    recovered.extra_documents = documents
    recovered.validate_proposition_documents()
    recovered._reset_runtime()
    from .supervisor import GuardedClustering, MutationPolicy

    recovered._recovery_clustering = GuardedClustering(
        recovered.assignments.copy(),
        new_cluster_id=recovered.next_cluster_id,
        spikes_per_cluster=recovered._assignment_groups,
        policy=MutationPolicy(),
    )
    if not np.array_equal(recovered._recovery_clustering.spike_clusters, recovered.assignments):
        raise ValueError('Recovered native assignments disagree with semantic projection.')
    recovered._certified_assignments = recovered.assignments
    recovered._certified_native = recovered._recovery_clustering._spike_clusters
    return recovered
