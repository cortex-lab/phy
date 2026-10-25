"""Semantic bounded curation and crash-consistent, immutable save generations."""

import copy
import hashlib
import json
import os
import shutil
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .metadata import POLICY, complete_schema, validate_schema
from .source import _time_base, hash_json

MANIFEST = '_phy_curation_manifest.json'
BASE_ASSIGNMENTS = '_phy_spike_clusters_base.npy'
BOOLEAN_CAPABILITIES = (
    'merge',
    'label',
    'split',
    'stored_waveforms',
    'raw_traces',
    'normalized_correlograms',
)


def _json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    ).encode()


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open(encoding='utf8') as stream:
        return json.load(stream)


def _write_json(path, value):
    with Path(path).open('wb') as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _safe_file(root, name):
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or name in ('', '.', '..')
        or any(c in name for c in ('\\', '\0', '\n', '\r'))
    ):
        raise ValueError('Manifest/checkpoint filename must be a plain local filename.')
    path = Path(root) / name
    if path.is_symlink():
        raise ValueError('Manifest/checkpoint files must not be symbolic links.')
    return path


def _validate_digest(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in '0123456789abcdef' for c in value)
    ):
        raise ValueError('Invalid SHA-256 digest.')
    return value


@contextmanager
def _commit_lock(root):
    """Hold a process-safe OS lock; process exit releases it automatically."""
    with _safe_file(root, 'commit.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt

            # Windows byte-range locking requires at least one byte in the file.
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _current_pointer(root):
    try:
        return _read_json(root / 'current.json')
    except FileNotFoundError:
        return None


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer.')
    return int(value)


def _validate_capabilities(manifest):
    capabilities = manifest.get('capabilities', {})
    if not isinstance(capabilities, dict):
        raise ValueError('Bounded capabilities must be a mapping.')
    if (
        set(capabilities)
        - set(BOOLEAN_CAPABILITIES)
        - {
            'full_attributes',
            'predicate_split_attributes',
        }
    ):
        raise ValueError('Unsupported bounded capability declaration.')
    result = {}
    for name in BOOLEAN_CAPABILITIES:
        value = capabilities.get(name, False)
        if type(value) is not bool:
            raise ValueError(f'Bounded capability {name!r} must be a boolean.')
        result[name] = value
    if result['split']:
        raise ValueError('General selected-row splits are unsupported in bounded mode.')
    for name in ('full_attributes', 'predicate_split_attributes'):
        values = capabilities.get(name, [])
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f'Bounded capability {name!r} must be a unique string list.')
        result[name] = list(values)
    if set(result['predicate_split_attributes']) - {'amplitudes.npy'}:
        raise ValueError(
            'Unsupported bounded predicate attribute; only raw amplitudes are supported.'
        )
    return result


def _field_name(field):
    if (
        not isinstance(field, str)
        or not field
        or field in ('.', '..')
        or any(char in field for char in ('/', '\\', '\0', '\n', '\r', '\t'))
    ):
        raise ValueError('Metadata fields must be nonempty safe local field names.')
    return field


def normalize_metadata(metadata):
    """Return JSON-compatible metadata with integer cluster keys."""
    if not isinstance(metadata, dict):
        raise ValueError('Cluster metadata must be a field-to-cluster mapping.')
    result = {}
    for field, values in metadata.items():
        _field_name(field)
        if not isinstance(values, dict):
            raise ValueError('Invalid cluster metadata field.')
        result[field] = {}
        for key, value in values.items():
            try:
                cluster = int(key)
            except (ValueError, TypeError) as error:
                raise ValueError('Invalid metadata cluster ID.') from error
            if str(cluster) != str(key) or cluster < 0:
                raise ValueError('Invalid metadata cluster ID.')
            _json_bytes(value)
            result[field][cluster] = copy.deepcopy(value)
    return result


def validate_predicate(predicate):
    """Validate a threshold on a canonical full-source scalar attribute."""
    if not isinstance(predicate, dict) or set(predicate) != {
        'attribute',
        'operator',
        'threshold',
        'nonfinite',
    }:
        raise ValueError('Threshold requires attribute/operator/threshold/nonfinite.')
    if predicate['attribute'] not in ('amplitudes.npy', 'spike_templates.npy'):
        raise ValueError('Unsupported full-source threshold attribute.')
    if predicate['operator'] not in ('<', '<=', '>', '>='):
        raise ValueError('Unsupported threshold comparator.')
    threshold = predicate['threshold']
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError('Threshold must be a finite number.')
    if not np.isfinite(threshold) or predicate['nonfinite'] not in ('keep', 'reject'):
        raise ValueError('Threshold needs a finite threshold and explicit nonfinite policy.')
    return copy.deepcopy(predicate)


def evaluate_predicate(values, predicate):
    values = np.asarray(values).reshape(-1)
    finite = np.isfinite(values)
    if predicate['nonfinite'] == 'reject' and not finite.all():
        raise ValueError('Nonfinite source values violate threshold policy.')
    operators = {'<': np.less, '<=': np.less_equal, '>': np.greater, '>=': np.greater_equal}
    return finite & operators[predicate['operator']](values, predicate['threshold'])


def interpret_events(unit_ids, metadata, events):
    """Validate event intent and resolve the final catalog and metadata."""
    active = set(map(int, unit_ids))
    if not active or any(i < 0 for i in active):
        raise ValueError('Full source unit catalog must contain nonnegative IDs.')
    allocated = set(active)
    metadata = normalize_metadata(metadata)
    if not isinstance(events, list):
        raise ValueError('Semantic journal must be an ordered event list.')
    next_id = max(active) + 1
    for event_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get('version') not in (1, 2):
            raise ValueError('Unsupported semantic journal event version.')
        operation = event.get('operation')
        if operation == 'metadata_policy':
            if (
                set(event) != {'version', 'operation', 'policy', 'schema'}
                or event['version'] != 2
                or event['policy'] != POLICY
            ):
                raise ValueError('Malformed metadata policy migration event.')
            validate_schema(event['schema'])
            # Old resolved values have no reliable curator intent. Preserve only
            # explicit label events on the current membership scope.
            derived = set()
            labels = {}
            for previous in events[:event_index]:
                op = previous['operation']
                if op == 'merge':
                    derived.add(previous['target'])
                elif op == 'threshold_split':
                    derived.update(previous['children'])
                elif op == 'label':
                    for unit in previous['ids']:
                        labels[(previous['field'], unit)] = previous['value']
            for field, values in metadata.items():
                for unit in derived & active:
                    values[unit] = copy.deepcopy(labels.get((field, unit)))
        elif operation == 'merge':
            if set(event) != {'version', 'operation', 'parents', 'target', 'metadata'}:
                raise ValueError('Malformed merge event.')
            parents = [_integer(i, 'Parent ID') for i in event['parents']]
            target = _integer(event['target'], 'Target ID')
            if target > np.iinfo(np.int32).max:
                raise ValueError('Target cluster ID exceeds the current Phy loader namespace.')
            if len(parents) < 2 or len(set(parents)) != len(parents):
                raise ValueError('Merge requires at least two distinct complete units.')
            if not set(parents) <= active or target in allocated or target < next_id:
                raise ValueError('Merge references inactive units or colliding target ID.')
            resolved = event['metadata']
            if not isinstance(resolved, dict) or any(not isinstance(f, str) for f in resolved):
                raise ValueError('Merge metadata must contain resolved field values.')
            if event['version'] == 2:
                expected = dict.fromkeys(set(metadata) | {'group'})
                if resolved != expected:
                    raise ValueError('New membership must clear producer metadata and quality.')
            _json_bytes(resolved)
            for field in resolved:
                _field_name(field)
            active.difference_update(parents)
            active.add(target)
            allocated.add(target)
            next_id = target + 1
            for values in metadata.values():
                for parent in parents:
                    values.pop(parent, None)
            for field, value in resolved.items():
                metadata.setdefault(field, {})[target] = copy.deepcopy(value)
        elif operation == 'label':
            if set(event) != {'version', 'operation', 'field', 'ids', 'value'}:
                raise ValueError('Malformed label event.')
            ids = [_integer(i, 'Label ID') for i in event['ids']]
            field = event['field']
            _field_name(field)
            if not ids or len(set(ids)) != len(ids) or not set(ids) <= active:
                raise ValueError('Label references inactive units or invalid metadata field.')
            _json_bytes(event['value'])
            values = metadata.setdefault(field, {})
            for cluster in ids:
                values[cluster] = copy.deepcopy(event['value'])
        elif operation == 'threshold_split':
            if set(event) != {
                'version',
                'operation',
                'parent',
                'children',
                'predicate',
                'metadata',
            }:
                raise ValueError('Malformed threshold split event.')
            parent = _integer(event['parent'], 'Parent ID')
            children = [_integer(i, 'Child ID') for i in event['children']]
            if any(child > np.iinfo(np.int32).max for child in children):
                raise ValueError('Child cluster ID exceeds the current Phy loader namespace.')
            validate_predicate(event['predicate'])
            if parent not in active or len(children) != 2 or len(set(children)) != 2:
                raise ValueError('Threshold split requires an active parent and two children.')
            if any(i in allocated or i < next_id for i in children):
                raise ValueError('Threshold split child ID collision.')
            resolved = normalize_metadata(event['metadata'])
            if event['version'] == 2:
                expected = {field: dict.fromkeys(children) for field in set(metadata) | {'group'}}
                if resolved != expected:
                    raise ValueError('New membership must clear producer metadata and quality.')
            if any(not set(v) <= set(children) for v in resolved.values()):
                raise ValueError('Split metadata must refer only to explicit children.')
            active.remove(parent)
            active.update(children)
            allocated.update(children)
            next_id = max(children) + 1
            for values in metadata.values():
                values.pop(parent, None)
            for field, values in resolved.items():
                metadata.setdefault(field, {}).update(values)
        else:
            raise ValueError(f'Unsupported semantic operation: {operation!r}.')
    return active, metadata, next_id


def project_assignments(assignments, events, attributes=None):
    """Interpret a validated journal against a bounded array or full-source batch."""
    output = np.asarray(assignments, dtype=np.int64).reshape(-1).copy()
    attributes = attributes or {}
    for event in events:
        if event['operation'] == 'merge':
            output[np.isin(output, event['parents'])] = event['target']
        elif event['operation'] == 'threshold_split':
            predicate = event['predicate']
            attribute = predicate['attribute']
            if attribute not in attributes:
                raise ValueError(f'Full-source threshold attribute is unavailable: {attribute}.')
            mask = output == event['parent']
            values = np.asarray(attributes[attribute]).reshape(-1)
            if len(values) != len(output):
                raise ValueError('Threshold attribute is not a scalar per-spike array.')
            selected = np.zeros(len(output), dtype=bool)
            selected[mask] = evaluate_predicate(values[mask], predicate)
            output[mask & ~selected] = event['children'][0]
            output[mask & selected] = event['children'][1]
    return output


class BoundedSession:
    """Bounded semantic state; reopen establishes a new undo baseline.

    Snapshot/restore integrate with the controller's undo stack. Saving publishes
    one pointer after every generation file has been flushed and verified.
    """

    def __init__(self, dataset_path):
        self.global_ids = None
        self.base_assignments = None
        self.attributes = {}
        try:
            self._initialize(dataset_path)
        except BaseException:
            self.close()
            raise

    def _initialize(self, dataset_path):
        self.path = Path(dataset_path)
        if self.path.is_file():
            self.path = self.path.parent
        self.manifest = _read_json(self.path / MANIFEST)
        manifest = self.manifest
        required = {
            'source',
            'sample_rate',
            'num_samples',
            'valid_intervals',
            'num_spikes',
            'bounded_num_spikes',
            'unit_ids',
            'counts',
            'next_cluster_id',
            'global_spike_ids_file',
            'windows_file',
        }
        if manifest.get('format') != 'phy_bounded_curation' or manifest.get('format_version') != 1:
            raise ValueError('Unsupported bounded curation manifest version.')
        if not required <= set(manifest):
            raise ValueError(
                'Incomplete bounded curation manifest; regenerate with prepare-large.'
            )
        source = manifest['source']
        if not isinstance(source, dict) or not {'identity', 'revision', 'path'} <= set(source):
            raise ValueError('Bounded manifest needs source identity and assignment revision.')
        if not isinstance(source['identity'], dict) or not isinstance(source['revision'], str):
            raise ValueError('Invalid source identity or revision.')
        identity = source['identity']
        identity_fields = {'format_version', 'acquisition', 'time_base', 'arrays', 'sha256'}
        if (
            not identity_fields <= set(identity)
            or identity.get('format_version') != 1
            or identity['sha256']
            != hash_json({k: v for k, v in identity.items() if k != 'sha256'})
            or len(source['revision']) != 64
            or any(c not in '0123456789abcdef' for c in source['revision'])
        ):
            raise ValueError('Invalid source coordinate identity or assignment revision digest.')
        n = _integer(manifest['bounded_num_spikes'], 'Bounded spike count')
        full_n = _integer(manifest['num_spikes'], 'Full spike count')
        samples = _integer(manifest['num_samples'], 'Recording sample count')
        if not n or n > full_n or not samples or not np.isfinite(manifest['sample_rate']):
            raise ValueError('Invalid bounded counts or recording duration.')
        if manifest['sample_rate'] <= 0:
            raise ValueError('Sampling rate must be positive.')
        expected_time_base = {
            'sample_rate': manifest['sample_rate'],
            'num_samples': samples,
            'valid_intervals': manifest['valid_intervals'],
        }
        if manifest.get('time_base') is not None:
            contract = _time_base(manifest['time_base'])
            if contract['ticks_per_second'] != manifest['sample_rate']:
                raise ValueError('Bounded tick frequency disagrees with sampling rate.')
            expected_time_base['contract'] = contract
            timestamps_hash = identity['time_base'].get('timestamps_sha256')
            if timestamps_hash is not None:
                if (
                    not isinstance(timestamps_hash, str)
                    or len(timestamps_hash) != 64
                    or any(c not in '0123456789abcdef' for c in timestamps_hash)
                ):
                    raise ValueError('Invalid source timestamps digest.')
                expected_time_base['timestamps_sha256'] = timestamps_hash
        if identity['time_base'] != expected_time_base:
            raise ValueError('Bounded time base disagrees with immutable source identity.')
        self.capabilities = _validate_capabilities(manifest)
        self.unit_ids = [_integer(i, 'Unit ID') for i in manifest['unit_ids']]
        if any(unit > np.iinfo(np.int32).max for unit in self.unit_ids):
            raise ValueError('Unit IDs exceed the current Phy loader namespace.')
        if not self.unit_ids or len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError('Full unit catalog must contain distinct unit IDs.')
        counts = np.asarray(manifest['counts'])
        if (
            counts.shape != (len(self.unit_ids),)
            or counts.dtype.kind not in 'iu'
            or np.any(counts < 0)
            or sum(map(int, counts)) != full_n
        ):
            raise ValueError('Full unit counts do not match the source spike count/catalog.')
        from phy.cluster._propositions import MergePropositionCatalog

        MergePropositionCatalog((), (), retired_units=manifest.get('retired_units', {}))
        if set(manifest.get('retired_units', {})) & set(map(str, self.unit_ids)):
            raise ValueError('Retired units cannot remain in the active source catalog.')
        self.base_next_cluster_id = _integer(manifest['next_cluster_id'], 'Next cluster ID')
        if self.base_next_cluster_id <= max(self.unit_ids):
            raise ValueError('Cluster allocator collides with the complete source catalog.')
        intervals = np.asarray(manifest['valid_intervals'])
        if (
            intervals.ndim != 2
            or intervals.shape[1] != 2
            or not len(intervals)
            or intervals.dtype.kind not in 'iu'
        ):
            raise ValueError('Valid acquisition intervals must be integer [start, end) pairs.')
        if (
            np.any(intervals[:, 0] < 0)
            or np.any(intervals[:, 1] > samples)
            or np.any(intervals[:, 1] <= intervals[:, 0])
            or np.any(intervals[1:, 0] < intervals[:-1, 1])
        ):
            raise ValueError('Invalid or overlapping acquisition intervals.')
        self.global_ids = np.load(
            _safe_file(self.path, manifest['global_spike_ids_file']), mmap_mode='r'
        )
        if self.global_ids.dtype != np.int64 or self.global_ids.shape != (n,):
            raise ValueError('Global spike IDs must be a one-dimensional int64 mapping.')
        if (
            np.any(self.global_ids < 0)
            or np.any(self.global_ids >= full_n)
            or np.any(self.global_ids[1:] <= self.global_ids[:-1])
        ):
            raise ValueError('Global spike IDs must be sorted, unique source row indices.')
        windows = np.load(_safe_file(self.path, manifest['windows_file']), mmap_mode='r')
        if (
            windows.ndim != 2
            or windows.shape[1] != 2
            or not len(windows)
            or windows.dtype != np.int64
        ):
            raise ValueError('Evidence windows must be int64 sample intervals.')
        if (
            np.any(windows[:, 0] < 0)
            or np.any(windows[:, 1] > samples)
            or np.any(windows[:, 1] <= windows[:, 0])
            or np.any(windows[1:, 0] < windows[:-1, 1])
        ):
            raise ValueError('Invalid or overlapping evidence windows.')
        for start, stop in windows:
            if not any(left <= start and stop <= right for left, right in intervals):
                raise ValueError('Evidence windows cross unsupported acquisition gaps.')
        for name, digest in manifest.get('bounded_files', {}).items():
            _validate_digest(digest)
            if _hash_file(_safe_file(self.path, name)) != digest:
                raise ValueError(f'Bounded package integrity check failed: {name}.')
        hashed_files = manifest.get('bounded_files', {})
        if not {
            BASE_ASSIGNMENTS,
            manifest['global_spike_ids_file'],
            manifest['windows_file'],
            'spike_times.npy',
        } <= set(hashed_files):
            raise ValueError('Bounded manifest lacks required immutable array digests.')
        self.base_assignments = np.load(self.path / BASE_ASSIGNMENTS, mmap_mode='r')
        if (
            self.base_assignments.shape != (n,)
            or self.base_assignments.dtype.kind not in 'iu'
            or not set(np.unique(self.base_assignments)) <= set(self.unit_ids)
        ):
            raise ValueError('Immutable bounded assignments do not match the full catalog.')
        self.metadata_schema = validate_schema(
            manifest.get('metadata_schema', {'version': 1, 'tables': []})
        )
        if manifest.get('metadata_policy') not in (None, POLICY):
            raise ValueError('Unsupported bounded metadata policy.')
        self.base_metadata = normalize_metadata(
            manifest.get('base_metadata', manifest.get('metadata', {}))
        )
        if any(not set(values) <= set(self.unit_ids) for values in self.base_metadata.values()):
            raise ValueError('Base metadata refers to units outside the full source catalog.')
        bounded_times = np.load(self.path / 'spike_times.npy', mmap_mode='r').reshape(-1)
        if (
            bounded_times.shape != (n,)
            or bounded_times.dtype.kind not in 'iu'
            or np.any(bounded_times[1:] < bounded_times[:-1])
        ):
            raise ValueError('Bounded spike sample coordinates have invalid alignment/order.')
        window_ids = np.searchsorted(windows[:, 0], bounded_times, side='right') - 1
        if np.any(window_ids < 0) or np.any(bounded_times >= windows[window_ids, 1]):
            raise ValueError('Bounded spikes lie outside the recorded evidence windows.')
        self.attributes = {}
        for name in ('amplitudes.npy', 'spike_templates.npy'):
            if (self.path / name).exists():
                self.attributes[name] = np.load(self.path / name, mmap_mode='r')
        for name in self.capabilities['predicate_split_attributes']:
            array = self.attributes.get(name)
            if (
                array is None
                or name not in identity['arrays']
                or not (array.ndim == 1 or array.shape[1:] == (1,))
                or len(array) != n
                or array.dtype.kind not in 'iuf'
            ):
                raise ValueError(
                    f'Bounded predicate attribute is not available as a source scalar: {name}.'
                )
        self.assignments = np.asarray(self.base_assignments, dtype=np.int64).copy()
        self.metadata = copy.deepcopy(self.base_metadata)
        self.events = []
        self.extra_documents = {}
        if 'proposition_documents' not in manifest and any(
            (self.path / name).exists() for name in ('curation.json', 'curation_review.json')
        ):
            raise ValueError(
                'Legacy bounded propositions have no preparation binding; reprepare the source.'
            )
        self.proposition_documents = manifest.get('proposition_documents', {})
        if not isinstance(self.proposition_documents, dict) or set(self.proposition_documents) - {
            'curation.json',
            'curation_review.json',
        }:
            raise ValueError('Invalid bounded proposition document inventory.')
        for name, item in self.proposition_documents.items():
            expected_name = (
                '_phy_curation_review_base.json' if name == 'curation_review.json' else name
            )
            if (
                not isinstance(item, dict)
                or set(item) != {'filename', 'sha256'}
                or item['filename'] != expected_name
                or hashed_files.get(expected_name) != item['sha256']
            ):
                raise ValueError('Proposition documents must have immutable preparation digests.')
            _validate_digest(item['sha256'])
            value = _read_json(_safe_file(self.path, expected_name))
            if not isinstance(value, dict):
                raise ValueError('Proposition documents must contain JSON objects.')
            if name == 'curation_review.json':
                self.extra_documents[name] = value
        if (
            'curation.json' in self.proposition_documents
            and 'curation_review.json' not in self.extra_documents
        ):
            from phy.cluster._propositions import encode_review_mapping

            self.extra_documents['curation_review.json'] = encode_review_mapping(
                {}, source_sha256=self.proposition_documents['curation.json']['sha256']
            )
        self.validate_proposition_documents()
        self.next_cluster_id = self.base_next_cluster_id
        self.checkpoint_path = None
        self._expected_pointer = None
        pointer_path = self.path / '.phy-curation' / 'current.json'
        if pointer_path.exists():
            self._load_checkpoint(pointer_path)
        if self.metadata_policy is None:
            warnings.warn(
                'Legacy bounded metadata preserves inherited values; use '
                'migrate_metadata() to adopt membership-scoped metadata.',
                UserWarning,
                stacklevel=2,
            )

    def validate_proposition_documents(self):
        """Keep producer input and its inherited ledger bound to preparation."""
        for name, item in self.proposition_documents.items():
            path = _safe_file(self.path, item['filename'])
            if not path.is_file() or _hash_file(path) != item['sha256']:
                raise ValueError(f'Bounded proposition document changed: {name}.')
        if 'proposition_documents' in self.manifest:
            for name in ('curation.json',):
                if name not in self.proposition_documents and (self.path / name).exists():
                    raise ValueError('Unbound bounded proposition document.')
        review = self.extra_documents.get('curation_review.json')
        has_review = 'curation_review.json' in self.extra_documents
        if has_review and 'proposition_documents' not in self.manifest:
            raise ValueError(
                'Legacy bounded reviews have no preparation binding; reprepare the source.'
            )
        if 'curation.json' in self.proposition_documents and not isinstance(review, dict):
            raise ValueError('Bounded proposition checkpoints require a review ledger mapping.')
        if has_review:
            from phy.cluster._propositions import decode_review_mapping

            source, _ = decode_review_mapping(review)
            expected = self.proposition_documents.get('curation.json', {}).get('sha256')
            if (
                source.get('filename') != 'curation.json'
                or expected is None
                or source.get('sha256') != expected
            ):
                raise ValueError('Bounded review source filename/hash does not match preparation.')

    @property
    def retired_units(self):
        """Cumulative retirement history, without retargeting source suggestions."""
        result = copy.deepcopy(self.manifest.get('retired_units', {}))
        # Events have already passed the complete-unit journal validator.
        interpret_events(self.unit_ids, self.base_metadata, self.events)
        for event in self.events:
            if event['operation'] == 'merge':
                for parent in event['parents']:
                    result[str(parent)] = {'operation': 'merge', 'replacements': [event['target']]}
            elif event['operation'] == 'threshold_split':
                result[str(event['parent'])] = {
                    'operation': 'threshold_split',
                    'replacements': list(event['children']),
                }
        return result

    def _validated_state(self, assignments, metadata, events, next_cluster_id):
        self._validate_event_capabilities(events)
        _, projected_metadata, minimum = interpret_events(
            self.unit_ids, self.base_metadata, events
        )
        projected = project_assignments(self.base_assignments, events, self.attributes)
        values = np.asarray(assignments)
        if values.dtype.kind not in 'iu' or values.shape != projected.shape:
            raise ValueError('Checkpoint assignments have invalid shape or dtype.')
        if not np.array_equal(values, projected):
            raise ValueError('Checkpoint assignments disagree with semantic journal projection.')
        if normalize_metadata(metadata) != projected_metadata:
            raise ValueError('Checkpoint metadata disagrees with semantic journal projection.')
        minimum = max(minimum, self.base_next_cluster_id)
        next_cluster_id = _integer(next_cluster_id, 'Next cluster ID')
        if next_cluster_id < minimum:
            raise ValueError('Checkpoint allocator collides with journal IDs.')
        return projected, projected_metadata, next_cluster_id

    def _load_checkpoint(self, pointer_path):
        pointer = _read_json(pointer_path)
        generation = _safe_file(pointer_path.parent / 'generations', pointer['generation'])
        descriptor = _read_json(generation / 'checkpoint.json')
        if _digest(descriptor) != pointer['sha256']:
            raise ValueError('Checkpoint commit pointer digest is invalid.')
        if descriptor.get('version') != 1 or descriptor.get('manifest_sha256') != _digest(
            self.manifest
        ):
            raise ValueError('Checkpoint belongs to a different bounded manifest.')
        for name, digest in descriptor['files'].items():
            _validate_digest(digest)
            if _hash_file(_safe_file(generation, name)) != digest:
                raise ValueError(f'Checkpoint corruption: {name}.')
        if not {'spike_clusters.npy', 'journal.json', 'metadata.json'} <= set(descriptor['files']):
            raise ValueError('Incomplete checkpoint file inventory.')
        events = _read_json(generation / 'journal.json')
        for event in events:
            if event.get('operation') == 'threshold_split':
                event['metadata'] = normalize_metadata(event['metadata'])
        metadata = _read_json(generation / 'metadata.json')
        assignments = np.load(generation / 'spike_clusters.npy', mmap_mode='r')
        values, metadata, next_id = self._validated_state(
            assignments, metadata, events, descriptor['next_cluster_id']
        )
        self.assignments, self.metadata = values, metadata
        self.events = events
        if 'metadata_policy' in descriptor and (
            descriptor['metadata_policy'] != self.metadata_policy
            or descriptor.get('metadata_schema') != self.effective_metadata_schema
            or descriptor.get('metadata_provenance') != self.metadata_provenance
        ):
            raise ValueError(
                'Checkpoint metadata policy/schema/provenance disagrees with journal.'
            )
        self.next_cluster_id = next_id
        self.checkpoint_path = generation
        self._expected_pointer = copy.deepcopy(pointer)
        documents = descriptor.get('extra_documents', [])
        if (
            not isinstance(documents, list)
            or any(not isinstance(name, str) for name in documents)
            or len(set(documents)) != len(documents)
            or not set(documents) <= set(descriptor['files'])
            or set(documents)
            & {'checkpoint.json', 'spike_clusters.npy', 'journal.json', 'metadata.json'}
        ):
            raise ValueError('Extra checkpoint documents must be in the hash inventory.')
        self.extra_documents = {
            name: _read_json(_safe_file(generation, name)) for name in documents
        }
        self.validate_proposition_documents()

    def snapshot(self):
        return {
            'assignments': self.assignments.copy(),
            'metadata': copy.deepcopy(self.metadata),
            'events': copy.deepcopy(self.events),
            'next_cluster_id': self.next_cluster_id,
        }

    def journal_state(self):
        """Return a compact history prefix without copying bounded spike arrays.

        Commands append event dictionaries and never modify earlier events. The
        tuple shares those immutable event records between history entries.
        """
        return {'events': tuple(self.events), 'next_cluster_id': self.next_cluster_id}

    def restore_journal(self, state):
        """Restore an active history prefix while preserving allocator high-water."""
        events = list(state['events'])
        self._validate_event_capabilities(events)
        _, metadata, minimum = interpret_events(self.unit_ids, self.base_metadata, events)
        next_id = _integer(state['next_cluster_id'], 'Next cluster ID')
        if next_id < max(minimum, self.base_next_cluster_id):
            raise ValueError('History allocator collides with semantic journal IDs.')
        assignments = project_assignments(self.base_assignments, events, self.attributes)
        self.assignments, self.metadata, self.events = assignments, metadata, events
        self.next_cluster_id = max(self.next_cluster_id, next_id)

    def close(self):
        """Release file-backed bounded readers independently of the GUI model."""
        for array in (self.global_ids, self.base_assignments, *self.attributes.values()):
            mapping = getattr(array, '_mmap', None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    def restore(self, snapshot):
        assignments, metadata, next_id = self._validated_state(**snapshot)
        self.assignments, self.metadata = assignments, metadata
        self.events = copy.deepcopy(snapshot['events'])
        self.next_cluster_id = next_id

    def _append(self, event):
        events = self.events + [event]
        self._validate_event_capabilities(events)
        _, metadata, next_id = interpret_events(self.unit_ids, self.base_metadata, events)
        assignments = project_assignments(self.base_assignments, events, self.attributes)
        self.assignments, self.metadata, self.events = assignments, metadata, events
        self.next_cluster_id = max(self.next_cluster_id, next_id)
        return copy.deepcopy(event)

    def _require_capability(self, operation, predicate=None):
        if operation == 'threshold_split':
            predicate = validate_predicate(predicate)
            if predicate['attribute'] not in self.capabilities['predicate_split_attributes']:
                raise ValueError(
                    'Predicate split capability is unavailable in this bounded package.'
                )
        elif not self.capabilities.get(operation, False):
            raise ValueError(
                f'{operation.capitalize()} capability is disabled in this bounded package.'
            )

    def _validate_event_capabilities(self, events):
        policy = self.manifest.get('metadata_policy') == POLICY
        for event in events:
            if isinstance(event, dict):
                operation = event.get('operation')
                if operation == 'metadata_policy':
                    if policy:
                        raise ValueError('Metadata policy migration has already been applied.')
                    policy = True
                elif operation in ('merge', 'label', 'threshold_split'):
                    if event.get('version') != (2 if policy else 1):
                        raise ValueError('Semantic event version disagrees with metadata policy.')
                    self._require_capability(operation, event.get('predicate'))

    @property
    def metadata_policy(self):
        return self._metadata_policy(self.events)

    def _metadata_policy(self, events):
        return (
            POLICY
            if (
                self.manifest.get('metadata_policy') == POLICY
                or any(e['operation'] == 'metadata_policy' for e in events)
            )
            else None
        )

    @property
    def effective_metadata_schema(self):
        return self._metadata_schema(self.events, self.metadata)

    def _metadata_schema(self, events, metadata):
        schema = self.metadata_schema
        for event in events:
            if event['operation'] == 'metadata_policy':
                schema = event['schema']
        return complete_schema(schema, metadata)

    @property
    def metadata_provenance(self):
        return self._metadata_provenance(self.events)

    def _metadata_provenance(self, events):
        result = copy.deepcopy(self.manifest.get('metadata_provenance', {'history': []}))
        metadata = copy.deepcopy(self.base_metadata)
        for index, event in enumerate(events):
            operation = event['operation']
            if operation in ('merge', 'threshold_split'):
                parents = event['parents'] if operation == 'merge' else [event['parent']]
                children = [event['target']] if operation == 'merge' else event['children']
                result.setdefault('history', []).append(
                    {
                        'operation': operation,
                        'source_revision': self.manifest['source']['revision'],
                        'event_index': index,
                        **(
                            {'predicate': copy.deepcopy(event['predicate'])}
                            if operation == 'threshold_split'
                            else {}
                        ),
                        'parents': parents,
                        'children': children,
                        'parent_metadata': {
                            field: {str(unit): values.get(unit) for unit in parents}
                            for field, values in metadata.items()
                        },
                        'policy': POLICY if event['version'] == 2 else 'legacy_inherited',
                    }
                )
            elif operation == 'label' or operation == 'metadata_policy':
                result.setdefault('history', []).append(copy.deepcopy(event))
            # Event metadata is already resolved and validated. Scan it once
            # instead of replaying every prefix of a potentially long journal.
            if operation in ('merge', 'threshold_split'):
                for values in metadata.values():
                    for parent in parents:
                        values.pop(parent, None)
                for field, resolved in event['metadata'].items():
                    values = {event['target']: resolved} if operation == 'merge' else resolved
                    metadata.setdefault(field, {}).update(copy.deepcopy(values))
            elif operation == 'label':
                values = metadata.setdefault(event['field'], {})
                for unit in event['ids']:
                    values[unit] = copy.deepcopy(event['value'])
            elif operation == 'metadata_policy':
                _, metadata, _ = interpret_events(
                    self.unit_ids, self.base_metadata, events[: index + 1]
                )
        return result

    def migrate_metadata(self, source_params=None):
        """Explicitly clear ambiguous inherited values; retain label journal intent.

        The original manifest and generations remain immutable. Saving commits a
        policy transition and schema in the same atomic generation as assignments.
        Reopen a supervisor after migrating an already-open GUI session.
        """
        if self.metadata_policy == POLICY:
            return None
        from .source import Source

        source_params = source_params or self.manifest.get('source', {}).get('path')
        if not source_params:
            raise ValueError('Migration requires source_params to recover original table names.')
        with Source(
            source_params,
            duration_seconds=self.manifest['num_samples'] / self.manifest['sample_rate'],
            valid_intervals=self.manifest['valid_intervals'],
        ) as source:
            if source.identity != self.manifest['source']['identity']:
                raise ValueError('Migration source identity does not match preparation.')
            if source.revision != self.manifest['source']['revision']:
                raise ValueError('Migration source revision does not match preparation.')
            if source.metadata != self.base_metadata:
                raise ValueError('Migration base metadata does not match source metadata.')
            schema = source.metadata_schema
        return self._append(
            {'version': 2, 'operation': 'metadata_policy', 'policy': POLICY, 'schema': schema}
        )

    def preflight_merge(self, parents, target, metadata=None):
        self._require_capability('merge')
        event = {
            'version': 2 if self.metadata_policy else 1,
            'operation': 'merge',
            'parents': [int(i) for i in parents],
            'target': _integer(target, 'Target ID'),
            'metadata': (
                dict.fromkeys(set(self.metadata) | {'group'})
                if self.metadata_policy
                else copy.deepcopy(metadata or {})
            ),
        }
        if event['target'] < self.next_cluster_id:
            raise ValueError('Merge target is below the complete catalog allocator.')
        interpret_events(self.unit_ids, self.base_metadata, self.events + [event])
        return event

    def record_merge(self, parents, target, metadata=None):
        return self._append(self.preflight_merge(parents, target, metadata))

    def preflight_label(self, field, ids, value):
        self._require_capability('label')
        complete_schema(self.effective_metadata_schema, {**self.metadata, field: {}})
        event = {
            'version': 2 if self.metadata_policy else 1,
            'operation': 'label',
            'field': field,
            'ids': [int(i) for i in ids],
            'value': copy.deepcopy(value),
        }
        interpret_events(self.unit_ids, self.base_metadata, self.events + [event])
        return event

    def record_label(self, field, ids, value):
        return self._append(self.preflight_label(field, ids, value))

    def record_threshold_split(self, parent, children, predicate, metadata=None):
        """Apply an explicitly enabled full-source scalar predicate."""
        self._require_capability('threshold_split', predicate)
        children = [int(i) for i in children]
        if any(i < self.next_cluster_id for i in children):
            raise ValueError('Threshold children collide with the complete catalog allocator.')
        return self._append(
            {
                'version': 2 if self.metadata_policy else 1,
                'operation': 'threshold_split',
                'parent': int(parent),
                'children': children,
                'predicate': validate_predicate(predicate),
                'metadata': (
                    {field: dict.fromkeys(children) for field in set(self.metadata) | {'group'}}
                    if self.metadata_policy
                    else normalize_metadata(metadata or {})
                ),
            }
        )

    def preflight_split(self, parent, selected_id, remainder_id, predicate, metadata=None):
        self._require_capability('threshold_split', predicate)
        children = [int(remainder_id), int(selected_id)]
        if any(i < self.next_cluster_id for i in children):
            raise ValueError('Threshold children collide with the complete catalog allocator.')
        event = {
            'version': 2 if self.metadata_policy else 1,
            'operation': 'threshold_split',
            'parent': int(parent),
            'children': children,
            'predicate': validate_predicate(predicate),
            'metadata': (
                {field: dict.fromkeys(children) for field in set(self.metadata) | {'group'}}
                if self.metadata_policy
                else normalize_metadata(metadata or {})
            ),
        }
        interpret_events(self.unit_ids, self.base_metadata, self.events + [event])
        project_assignments(self.base_assignments, self.events + [event], self.attributes)
        return event

    def record_split(self, parent, selected_id, remainder_id, predicate, metadata=None):
        return self._append(
            self.preflight_split(parent, selected_id, remainder_id, predicate, metadata)
        )

    def save(
        self,
        assignments=None,
        metadata=None,
        events=None,
        next_cluster_id=None,
        extra_documents=None,
    ):
        """Validate, flush a generation, then atomically replace its commit pointer."""
        self.validate_proposition_documents()
        assignments = self.assignments if assignments is None else assignments
        metadata = self.metadata if metadata is None else metadata
        events = self.events if events is None else events
        next_cluster_id = self.next_cluster_id if next_cluster_id is None else next_cluster_id
        values, metadata, next_id = self._validated_state(
            assignments, metadata, events, next_cluster_id
        )
        if extra_documents is not None and not isinstance(extra_documents, dict):
            raise ValueError('Extra checkpoint documents must be a filename-to-JSON mapping.')
        documents = copy.deepcopy(self.extra_documents)
        if extra_documents is not None:
            documents.update(copy.deepcopy(extra_documents))
        if not isinstance(documents, dict):
            raise ValueError('Extra checkpoint documents must be a filename-to-JSON mapping.')
        for name, value in documents.items():
            _safe_file(self.path, name)
            if name in ('checkpoint.json', 'spike_clusters.npy', 'journal.json', 'metadata.json'):
                raise ValueError('Extra document collides with a checkpoint state file.')
            _json_bytes(value)
        previous_documents = self.extra_documents
        try:
            self.extra_documents = documents
            self.validate_proposition_documents()
        finally:
            self.extra_documents = previous_documents
        root = self.path / '.phy-curation'
        generations = root / 'generations'
        generations.mkdir(parents=True, exist_ok=True)
        _fsync_directory(root)
        _fsync_directory(self.path)
        with _commit_lock(root):
            if _current_pointer(root) != self._expected_pointer:
                raise ValueError(
                    'Bounded curation checkpoint changed in another session; '
                    'reopen the dataset before saving to avoid losing its changes.'
                )
            return self._save_generation(generations, values, metadata, events, next_id, documents)

    def _save_generation(self, generations, values, metadata, events, next_id, documents):
        root = generations.parent
        identifier = uuid.uuid4().hex
        stage = generations / f'.{identifier}.tmp'
        final = generations / identifier
        stage.mkdir()
        pointer_tmp = root / f'.current-{identifier}.tmp'
        try:
            np.save(stage / 'spike_clusters.npy', values)
            # Windows _commit (os.fsync) requires a writable descriptor.
            with (stage / 'spike_clusters.npy').open('r+b') as stream:
                os.fsync(stream.fileno())
            _write_json(stage / 'journal.json', events)
            _write_json(stage / 'metadata.json', metadata)
            for name, value in documents.items():
                _write_json(stage / name, value)
            descriptor = {
                'version': 1,
                'manifest_sha256': _digest(self.manifest),
                'next_cluster_id': next_id,
                'metadata_policy': self._metadata_policy(events),
                'metadata_schema': self._metadata_schema(events, metadata),
                'metadata_provenance': self._metadata_provenance(events),
                'files': {
                    name: _hash_file(stage / name)
                    for name in ('spike_clusters.npy', 'journal.json', 'metadata.json', *documents)
                },
                'extra_documents': sorted(documents),
            }
            _write_json(stage / 'checkpoint.json', descriptor)
            _fsync_directory(stage)
            os.replace(stage, final)
            _fsync_directory(generations)
            pointer = {'generation': identifier, 'sha256': _digest(descriptor)}
            _write_json(pointer_tmp, pointer)
            self.validate_proposition_documents()
            os.replace(pointer_tmp, root / 'current.json')
            _fsync_directory(root)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            pointer_tmp.unlink(missing_ok=True)
            raise
        self.assignments, self.metadata, self.events = values, metadata, copy.deepcopy(events)
        self.next_cluster_id, self.checkpoint_path = next_id, final
        self.extra_documents = documents
        self._expected_pointer = pointer
        return final
