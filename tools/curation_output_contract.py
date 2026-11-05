"""Source-grounded file contract for independent curation acceptance.

This checker does not call the bounded assignment projector or metadata event
interpreter. It checks native files, full row identities, and cumulative action
semantics against the original source and explicit final curator expectations.
"""

import ast
import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

from phy.apps.large_recording.source import Source


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _json_hash(value, newline=False):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256((text + ('\n' if newline else '')).encode()).hexdigest()


def _cell(value):
    if value is None or value == '':
        return ''
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return str(value)


def _table(path):
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=',' if path.suffix == '.csv' else '\t')
        columns = reader.fieldnames or []
        if (
            len(columns) < 2
            or columns.count('cluster_id') != 1
            or len(set(columns)) != len(columns)
        ):
            raise ValueError(f'Invalid metadata columns: {path.name}')
        values = {field: {} for field in columns if field != 'cluster_id'}
        seen = set()
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed metadata row: {path.name}')
            unit = int(row['cluster_id'])
            if unit < 0 or unit in seen:
                raise ValueError(f'Invalid or duplicate metadata unit: {path.name}')
            seen.add(unit)
            for field in values:
                values[field][unit] = row[field]
    return columns, values


def _native_metadata(path, source, fields, active_ids):
    """Native canonical tables take precedence only over proven source aliases."""
    tables = {
        p.name: _table(p)
        for p in sorted([*path.glob('cluster_*.tsv'), *path.glob('cluster_*.csv')])
        if p.stem != 'cluster_info'
    }
    source_tables = {t['filename']: t for t in source.metadata_schema['tables']}
    canonical, aliases = {}, []
    for field in fields:
        name = f'cluster_{field}.tsv'
        if name in tables and tables[name][0] == ['cluster_id', field]:
            canonical[field] = tables[name][1][field]
    result = {}
    active_ids = set(active_ids)

    def current_rows(name, field, rows):
        # Native loading omits completely blank imported columns, leaving their
        # original files untouched even after some original IDs are retired.
        # Recognize only an exact byte-pinned, entirely blank source column.
        if (
            set(rows) - active_ids
            and name in source_tables
            and all(not _cell(v) for v in source.metadata.get(field, {}).values())
            and _hash(path / name) == _hash(source.path / name)
        ):
            aliases.append(
                {'filename': name, 'field': field, 'policy': 'byte-pinned blank source column'}
            )
            return {u: v for u, v in rows.items() if u in active_ids}
        return rows

    for name, (columns, values) in tables.items():
        if len(values) == 1:
            field = next(iter(values))
            if name == f'cluster_{field}.tsv' and field in canonical:
                if field not in fields:
                    raise ValueError(f'Unexpected native metadata field: {field}')
                result[field] = current_rows(name, field, canonical[field])
                continue
        descriptor = source_tables.get(name)
        if descriptor is None or columns != descriptor['columns']:
            raise ValueError(f'Unsupported ordinary metadata layout: {name}')
        if _hash(path / name) != _hash(source.path / name):
            raise ValueError(f'Ordinary source alias changed: {name}')
        for field, rows in values.items():
            if field in canonical:
                aliases.append(
                    {'filename': name, 'field': field, 'policy': 'unchanged source alias'}
                )
            elif field in result:
                raise ValueError(f'Ambiguous ordinary metadata field: {field}')
            else:
                result[field] = current_rows(name, field, rows)
    return result, aliases, set(tables)


def _parameters(path, source):
    values = {
        k: v for k, v in runpy.run_path(str(path / 'params.py')).items() if not k.startswith('__')
    }

    def parameter_code(filename):
        tree = ast.parse(filename.read_bytes())
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ('dat_path', 'num_samples', 'valid_intervals')
            )
        ]
        return ast.dump(tree, include_attributes=False)

    if parameter_code(path / 'params.py') != parameter_code(source.params_path):
        raise ValueError('Changed parameter Python outside managed overrides')
    allowed_added = {'num_samples', 'valid_intervals'}
    if set(values) - set(source.params) - allowed_added:
        raise ValueError('Unexpected added parameter keys')
    if set(source.params) - set(values):
        raise ValueError('Missing source parameter keys')
    for key, value in source.params.items():
        if key in ('dat_path', 'num_samples', 'valid_intervals'):
            continue
        if callable(value) or type(value).__name__ == 'module':
            continue  # Unchanged helper/import syntax was checked above.
        if not np.array_equal(values[key], value):
            raise ValueError(f'Changed source parameter: {key}')
    if values.get('num_samples', source.num_samples) != source.num_samples:
        raise ValueError('Changed recording extent')
    if not np.array_equal(
        values.get('valid_intervals', source.valid_intervals), source.valid_intervals
    ):
        raise ValueError('Changed acquisition exposure')

    def raw_paths(value, base):
        items = value if isinstance(value, (tuple, list)) else [value]
        return [
            str((base / str(p)).resolve())
            for p in items
            if p is not None and str(p).strip() not in ('', 'None')
        ]

    expected = raw_paths(source.params.get('dat_path'), source.path)
    actual = raw_paths(values.get('dat_path'), path)
    if actual != expected:
        raise ValueError('Changed raw data reference')


def _history_records(provenance):
    if not isinstance(provenance, dict) or not isinstance(provenance.get('history'), list):
        raise ValueError('Malformed metadata provenance/history')
    if any(
        not isinstance(record, dict) or not isinstance(record.get('operation'), str)
        for record in provenance['history']
    ):
        raise ValueError('Malformed metadata history record')
    return provenance['history']


def _predicate(value, arrays):
    if not isinstance(value, dict) or set(value) != {
        'attribute',
        'operator',
        'threshold',
        'nonfinite',
    }:
        raise ValueError('Malformed threshold predicate')
    threshold = value['threshold']
    if (
        value['attribute'] not in ('amplitudes.npy', 'spike_templates.npy')
        or value['attribute'] not in arrays
        or value['operator'] not in ('<', '<=', '>', '>=')
        or value['nonfinite'] not in ('keep', 'reject')
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
    ):
        raise ValueError('Unsupported threshold predicate schema')


def _event_schema(event, record, arrays):
    operation = event.get('operation')
    keys = {
        'merge': {'version', 'operation', 'parents', 'target', 'metadata'},
        'threshold_split': {'version', 'operation', 'parent', 'children', 'predicate', 'metadata'},
        'label': {'version', 'operation', 'field', 'ids', 'value'},
    }
    if (
        operation not in keys
        or set(event) != keys[operation]
        or type(event['version']) is not int
        or event['version'] != 2
    ):
        raise ValueError('Malformed current journal event schema')
    if operation == 'label':
        return
    metadata = event['metadata']
    # New memberships also introduce the built-in unreviewed group field.
    fields = set(record['parent_metadata']) | {'group'}
    if not isinstance(metadata, dict) or set(metadata) != fields:
        raise ValueError('Journal membership metadata field coverage disagrees')
    if operation == 'merge':
        valid = all(value is None for value in metadata.values())
    else:
        _predicate(event['predicate'], arrays)
        children = {str(u) for u in event['children']}
        valid = all(
            isinstance(values, dict)
            and set(values) == children
            and all(value is None for value in values.values())
            for values in metadata.values()
        )
    if not valid:
        raise ValueError('Journal membership metadata contradicts clearing policy')


def _history(source, curated, journal, manifest, expected_labels):
    """Track only action ancestry and explicit label intent, never spike projection."""
    original = _history_records(source.metadata_provenance)
    history = _history_records(curated.metadata_provenance)
    if history[: len(original)] != original:
        raise ValueError('Lost original metadata provenance')
    if set(curated.metadata_provenance) != set(source.metadata_provenance) or any(
        curated.metadata_provenance[k] != value
        for k, value in source.metadata_provenance.items()
        if k != 'history'
    ):
        raise ValueError('Changed non-history metadata provenance')
    inherited = (
        _history_records(manifest.get('metadata_provenance', {'history': []}))
        if manifest is not None
        else original
    )
    if history[: len(inherited)] != inherited or len(history) != len(inherited) + len(journal):
        raise ValueError('Cumulative metadata history disagrees with preparation/journal')
    for event_index, (event, record) in enumerate(
        zip(journal, history[len(inherited) :], strict=True)
    ):
        _event_schema(event, record, source.arrays)
        operation = event['operation']
        if operation != record['operation'] or event.get('version') != 2:
            raise ValueError('Journal operation/version disagrees with history')
        if operation == 'merge':
            valid = record['parents'] == event['parents'] and record['children'] == [
                event['target']
            ]
        elif operation == 'threshold_split':
            valid = (
                record['parents'] == [event['parent']]
                and record['children'] == event['children']
                and record['predicate'] == event['predicate']
            )
        elif operation == 'label':
            valid = record == event
        else:
            raise ValueError('Unsupported metadata policy transition in contract')
        if not valid:
            raise ValueError('Journal details disagree with cumulative history')
        if operation in ('merge', 'threshold_split'):
            parent_revision = (
                manifest['source']['revision'] if manifest is not None else source.revision
            )
            if record['event_index'] != event_index:
                raise ValueError('Membership provenance has wrong event index')
            if record['source_revision'] != parent_revision:
                raise ValueError('Membership provenance has wrong source revision')
    fields = set(source.metadata)
    if any(
        record['operation'] in ('merge', 'threshold_split') for record in history[len(original) :]
    ):
        fields.add('group')
    fields.update(
        record['field'] for record in history[len(original) :] if record['operation'] == 'label'
    )
    if any(set(values) - fields for values in expected_labels.values()):
        raise ValueError('Expected labels have no source field or recorded label intent')
    if set(curated.metadata) != fields:
        raise ValueError('Unexpected replay metadata fields without label intent')
    candidates = {
        int(u): {f: {_cell(source.metadata.get(f, {}).get(int(u)))} for f in fields}
        for u in source.unit_ids
    }
    resolved = {
        f: {int(u): _cell(v) for u, v in values.items()} for f, values in source.metadata.items()
    }
    active = set(candidates)
    next_id = source.next_cluster_id
    retired = dict(source.retired_units)
    explicit = {}
    for record in history[len(original) :]:
        operation = record['operation']
        id_lists = ('ids',) if operation == 'label' else ('parents', 'children')
        for key in id_lists:
            ids = record.get(key)
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(u) is not int or u < 0 for u in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError('Malformed unit IDs in metadata history')
        if operation == 'label':
            field, value = record['field'], _cell(record['value'])
            if field not in fields or not set(record['ids']) <= active:
                raise ValueError('History label has unknown field/unit')
            for unit in record['ids']:
                candidates[unit][field] = {value}
                explicit.setdefault(unit, {})[field] = value
                resolved.setdefault(field, {})[unit] = value
        elif operation in ('merge', 'threshold_split'):
            record_keys = {
                'operation',
                'source_revision',
                'event_index',
                'parents',
                'children',
                'parent_metadata',
                'policy',
            }
            if operation == 'threshold_split':
                record_keys.add('predicate')
            if set(record) != record_keys:
                raise ValueError('Unexpected membership provenance fields')
            parents, children = record['parents'], record['children']
            if (operation == 'merge' and len(parents) < 2) or (
                operation == 'threshold_split' and len(parents) != 1
            ):
                raise ValueError('Invalid membership parent cardinality')
            if operation == 'threshold_split':
                _predicate(record['predicate'], source.arrays)
            if (
                record.get('policy') != 'membership_v1'
                or len(set(parents)) != len(parents)
                or not set(parents) <= active
            ):
                raise ValueError('Unsupported or inactive membership history')
            if (
                len(children) != (1 if operation == 'merge' else 2)
                or len(set(children)) != len(children)
                or min(children) < next_id
            ):
                raise ValueError('Invalid allocated history identities')
            snapshot = {
                f: {str(u): values.get(u, '') for u in parents} for f, values in resolved.items()
            }
            recorded = {
                f: {str(u): _cell(v) for u, v in values.items()}
                for f, values in record['parent_metadata'].items()
            }
            if snapshot != recorded:
                raise ValueError('Parent metadata provenance disagrees with independent history')
            inherited = {f: set().union(*(candidates[u][f] for u in parents)) for f in fields}
            active.difference_update(parents)
            for parent in parents:
                retired[str(parent)] = {'operation': operation, 'replacements': children}
            for parent in parents:
                explicit.pop(parent, None)
                for values in resolved.values():
                    values.pop(parent, None)
            resolved.setdefault('group', {})
            for child in children:
                candidates[child] = {f: set(v) for f, v in inherited.items()}
                for values in resolved.values():
                    values[child] = ''
                active.add(child)
            next_id = max(children) + 1
        else:
            raise ValueError(f'Unsupported cumulative history operation: {operation}')
    if retired != curated.retired_units:
        raise ValueError('Retired unit lineage disagrees with independent history')
    if curated.next_cluster_id < next_id:
        raise ValueError('Allocation floor disagrees with cumulative history')
    if active != set(map(int, curated.unit_ids)):
        raise ValueError('Final catalog disagrees with independent action ancestry')
    if not set(expected_labels) <= active:
        raise ValueError('Explicit expectations refer to inactive units')
    for unit, values in explicit.items():
        for field, value in values.items():
            if expected_labels.get(unit, {}).get(field) != value:
                raise ValueError(
                    f'Missing or wrong independent final label expectation: {unit}.{field}'
                )
    for unit, values in expected_labels.items():
        for field, value in values.items():
            intended = explicit.get(unit, {}).get(
                field,
                _cell(source.metadata.get(field, {}).get(unit))
                if unit in set(map(int, source.unit_ids))
                else '',
            )
            if value != intended:
                raise ValueError('Independent expectation disagrees with recorded label intent')
    return fields, candidates, explicit, active, history[len(original) :]


def _assignment_oracle(source, curated, history, chunk_size):
    """Independently execute cumulative membership intent on every original row.

    Attribute values always come from the immutable original source. Only a
    chunk of assignments is copied, including when earlier rounds split a unit
    that is subsequently merged or split again. Labels have no row effect.
    """
    original = source.arrays['spike_clusters.npy'].reshape(-1)
    current = curated.arrays['spike_clusters.npy'].reshape(-1)
    membership = [r for r in history if r['operation'] != 'label']
    attributes = {}
    for record in membership:
        if record['operation'] != 'threshold_split':
            continue
        name = record['predicate']['attribute']
        values = source.arrays[name]
        if values.shape not in ((len(original),), (len(original), 1)):
            raise ValueError('Predicate attribute must be a scalar per source spike')
        attributes[name] = values.reshape(-1)
    if len(current) != len(original):
        raise ValueError('Assignment oracle requires complete source rows')
    for start in range(0, len(original), chunk_size):
        stop = start + chunk_size
        expected = np.array(original[start:stop], dtype=np.int64, copy=True)
        for record in membership:
            parents, children = record['parents'], record['children']
            if record['operation'] == 'merge':
                expected[np.isin(expected, parents)] = children[0]
                continue
            parent_rows = np.flatnonzero(expected == parents[0])
            predicate = record['predicate']
            values = attributes[predicate['attribute']][start:stop][parent_rows]
            finite = np.isfinite(values)
            if predicate['nonfinite'] == 'reject' and not finite.all():
                raise ValueError('Predicate rejects nonfinite values in source parent rows')
            threshold = predicate['threshold']
            operator = predicate['operator']
            if operator == '<':
                selected = values < threshold
            elif operator == '<=':
                selected = values <= threshold
            elif operator == '>':
                selected = values > threshold
            else:  # Schema has already restricted the fourth operator to >=.
                selected = values >= threshold
            # Nonfinite values always remain in the first (remainder) child.
            expected[parent_rows] = np.where(finite & selected, children[1], children[0])
        mismatch = np.flatnonzero(expected != current[start:stop])
        if len(mismatch):
            row = start + int(mismatch[0])
            raise ValueError(
                f'Assignment disagrees with cumulative source predicates at spike {row}'
            )
    return {
        'verified': True,
        'source_rows': len(original),
        'membership_events': len(membership),
        'threshold_splits': sum(r['operation'] == 'threshold_split' for r in membership),
    }


def _local_file(root, name):
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or any(c in name for c in ('\\', '\0', '\n', '\r'))
    ):
        raise ValueError('Unsafe checkpoint asset name')
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or linked checkpoint asset')
    return path


def _checkpoint(bounded, manifest, lineage, journal, right, curated, chunk_size):
    """Bind the exact saved generation, including allocator and retained rows."""
    root = Path(bounded)
    pointer = _json(root / '.phy-curation/current.json')
    generation = pointer['generation']
    if (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(c not in '0123456789abcdef' for c in generation)
    ):
        raise ValueError('Invalid checkpoint generation')
    directory = root / '.phy-curation/generations' / generation
    descriptor = _json(_local_file(directory, 'checkpoint.json'))
    if set(pointer) != {'generation', 'sha256'} or descriptor.get('version') != 1:
        raise ValueError('Unsupported checkpoint pointer/descriptor')
    if pointer['sha256'] != _json_hash(descriptor, newline=True) or descriptor[
        'manifest_sha256'
    ] != _json_hash(manifest, newline=True):
        raise ValueError('Checkpoint pointer/manifest binding changed')
    required = {'spike_clusters.npy', 'journal.json', 'metadata.json'}
    inventory = descriptor['files']
    if not required <= set(inventory) or not set(descriptor['extra_documents']) <= set(inventory):
        raise ValueError('Checkpoint inventory is incomplete')
    if {p.name for p in directory.iterdir()} != set(inventory) | {'checkpoint.json'}:
        raise ValueError('Unexpected checkpoint generation assets')
    for name, digest in inventory.items():
        if _hash(_local_file(directory, name)) != digest:
            raise ValueError('Checkpoint asset hash changed')
    if _json(directory / 'journal.json') != journal or _json(directory / 'metadata.json') != _json(
        right / '_phy_cluster_metadata.json'
    ):
        raise ValueError('Export differs from saved journal/metadata generation')
    for key in ('next_cluster_id', 'metadata_policy', 'metadata_provenance'):
        if descriptor[key] != lineage[key]:
            raise ValueError(f'Export differs from saved checkpoint: {key}')
    if descriptor['metadata_schema'] != curated.metadata_schema:
        raise ValueError('Export metadata schema differs from saved checkpoint')
    if 'curation_review.json' in inventory and _json(directory / 'curation_review.json') != _json(
        right / 'curation_review.json'
    ):
        raise ValueError('Export review differs from saved checkpoint')
    name = manifest['global_spike_ids_file']
    ids_path = _local_file(root, name)
    if _hash(ids_path) != manifest['bounded_files'][name]:
        raise ValueError('Retained row identity hash changed')
    ids = saved = None
    try:
        ids = np.load(ids_path, mmap_mode='r', allow_pickle=False)
        saved = np.load(directory / 'spike_clusters.npy', mmap_mode='r', allow_pickle=False)
        if (
            ids.dtype != np.int64
            or ids.shape != (manifest['bounded_num_spikes'],)
            or saved.shape != ids.shape
            or saved.dtype.kind not in 'iu'
        ):
            raise ValueError('Invalid saved retained assignment alignment')
        if np.any(ids < 0) or np.any(ids >= curated.n_spikes) or np.any(ids[1:] <= ids[:-1]):
            raise ValueError('Invalid retained row coordinates')
        full = curated.arrays['spike_clusters.npy'].reshape(-1)
        for start in range(0, len(ids), chunk_size):
            if not np.array_equal(
                full[ids[start : start + chunk_size]], saved[start : start + chunk_size]
            ):
                raise ValueError('Full replay disagrees with saved retained assignments')
    finally:
        for array in (ids, saved):
            if array is not None:
                array._mmap.close()
    return generation


def compare_file_contract(
    left, right, source_path, mapping, expected_labels=None, bounded_dir=None, chunk_size=100000
):
    """Return file report plus normalized native/replay label maps.

    ``source_path`` is the original immutable source even across multiple rounds.
    Current preparation binding is verified when ``bounded_dir`` is supplied;
    older revisions are recorded ancestry, not independently fetched snapshots.
    """
    left, right = Path(left), Path(right)
    source_path = Path(source_path)
    params = source_path if source_path.is_file() else source_path / 'params.py'
    expected_labels = {
        int(u): {f: _cell(v) for f, v in values.items()}
        for u, values in (expected_labels or {}).items()
    }
    source = curated = None
    try:
        source = Source(params, batch_size=chunk_size)
        curated = Source(right / 'params.py', batch_size=chunk_size)
        if source.identity != curated.identity:
            raise ValueError('Replay immutable coordinate identity differs from original source')
        lineage = _json(right / '_phy_curation_lineage.json')
        lineage_keys = {
            'format',
            'format_version',
            'coordinate_identity',
            'parent_revision',
            'journal_sha256',
            'revision',
            'preparation_sha256',
            'next_cluster_id',
            'unit_ids',
            'retired_units',
            'proposition_documents',
            'metadata_provenance',
            'metadata_policy',
        }
        if set(lineage) != lineage_keys:
            raise ValueError('Unexpected lineage fields')
        proposition_hashes = {
            name: _hash(right / name)
            for name in ('curation.json', 'curation_review.json')
            if (right / name).exists()
        }
        if lineage['proposition_documents'] != proposition_hashes:
            raise ValueError('Missing or wrong proposition lineage bindings')
        journal = _json(right / '_phy_curation_journal.json')
        if lineage.get('metadata_policy') != 'membership_v1':
            raise ValueError('Strict file contract requires membership_v1 replay')
        if lineage.get('journal_sha256') != _json_hash(journal, newline=True):
            raise ValueError('Journal hash disagrees with lineage')
        manifest = None
        if bounded_dir is not None:
            manifest = _json(Path(bounded_dir) / '_phy_curation_manifest.json')
            if (
                manifest.get('format') != 'phy_bounded_curation'
                or manifest.get('format_version') != 1
            ):
                raise ValueError('Unsupported bounded preparation format')
            if lineage.get('preparation_sha256') != _json_hash(manifest, newline=True):
                raise ValueError('Preparation hash disagrees with lineage')
            if (
                lineage.get('parent_revision') != manifest['source']['revision']
                or manifest['source']['identity'] != source.identity
            ):
                raise ValueError('Lineage parent/identity disagrees with preparation')
        elif lineage.get('parent_revision') != source.revision:
            raise ValueError('Later-round parent verification requires bounded_dir')
        fields, candidates, explicit, active, history = _history(
            source, curated, journal, manifest, expected_labels
        )
        saved_generation = None
        if bounded_dir is not None:
            saved_generation = _checkpoint(
                Path(bounded_dir), manifest, lineage, journal, right, curated, chunk_size
            )
        native, aliases, native_tables = _native_metadata(left, source, fields, mapping.keys())
        if set(native) != fields:
            raise ValueError('Missing native metadata field/table coverage')
        replay_metadata = {
            f: {int(u): _cell(v) for u, v in values.items()}
            for f, values in curated.metadata.items()
        }
        mapped = {int(a): int(b) for a, b in mapping.items()}
        reverse = {b: a for a, b in mapped.items()}
        if len(reverse) != len(mapped) or not set(reverse) <= active:
            raise ValueError('Partition mapping disagrees with final catalog')
        untouched = set(map(int, source.unit_ids)) & active
        assignment_oracle = _assignment_oracle(source, curated, history, chunk_size)
        differences = []
        for field in fields:
            for unit in active:
                value = replay_metadata.get(field, {}).get(unit, '')
                expected = expected_labels.get(unit, {}).get(
                    field,
                    _cell(source.metadata.get(field, {}).get(unit)) if unit in untouched else '',
                )
                if value != expected:
                    raise ValueError(
                        f'Replay metadata violates label/clearing policy: {unit}.{field}'
                    )
                if unit not in reverse:
                    # Empty source units are deliberately not silently matched.
                    raise ValueError(f'Final metadata unit has no mapped source rows: {unit}')
                left_id = reverse[unit]
                native_value = native.get(field, {}).get(left_id, '')
                allowed = (
                    {expected}
                    if field in expected_labels.get(unit, {}) or unit in untouched
                    else candidates[unit][field] | {''}
                )
                if native_value not in allowed:
                    raise ValueError(
                        f'Ordinary metadata has unsupported ancestor value: {left_id}.{field}'
                    )
                if native_value != value:
                    differences.append(
                        {
                            'field': field,
                            'left_id': left_id,
                            'right_id': unit,
                            'left': native_value,
                            'right': value,
                            'policy': 'native ancestor inheritance versus cleared new membership',
                        }
                    )
            if set(native.get(field, {})) - set(mapped):
                raise ValueError(f'Unmapped ordinary metadata units: {field}')
        for path in (left, right):
            _parameters(path, source)
        immutable = set(source.arrays) - {'spike_clusters.npy'}
        immutable.update(
            n
            for n in ('_phy_spikes_subset.manifest.json', 'timestamps.json', 'curation.json')
            if (source.path / n).exists()
        )
        for name in sorted(immutable):
            expected_hash = _hash(source.path / name)
            for path in (left, right):
                if not (path / name).is_file() or _hash(path / name) != expected_hash:
                    raise ValueError(f'Immutable asset missing or changed: {path.name}/{name}')
        # Source layouts are authoritative for replay; only new label tables may be added.
        source_schema = {t['filename']: t for t in source.metadata_schema['tables']}
        right_schema = {t['filename']: t for t in curated.metadata_schema['tables']}
        for name, table in source_schema.items():
            if right_schema.get(name) != table:
                raise ValueError(f'Replay source metadata layout changed: {name}')
        for name, table in right_schema.items():
            if name not in source_schema:
                non_id = [f for f in table['columns'] if f != 'cluster_id']
                if (
                    len(non_id) != 1
                    or name != f'cluster_{non_id[0]}.tsv'
                    or table['columns'] != ['cluster_id', non_id[0]]
                    or non_id[0] in source.metadata
                ):
                    raise ValueError(f'Unexpected replay metadata table: {name}')
        for path in (left, right):
            review = path / 'curation_review.json'
            base_review = (
                _json(source.path / 'curation_review.json')
                if (source.path / 'curation_review.json').exists()
                else {'reviews': {}}
            )
            if review.exists():
                from phy.cluster._propositions import decode_review_mapping

                doc = _json(review)
                decode_review_mapping(doc)
                if (
                    doc.get('source', {}).get('filename') != 'curation.json'
                    or doc.get('source', {}).get('sha256') != _hash(source.path / 'curation.json')
                    or doc['reviews'] != base_review['reviews']
                ):
                    raise ValueError(
                        'Review binding or inherited decisions changed without expectations'
                    )
            elif base_review['reviews'] or (
                path == right and (source.path / 'curation.json').exists()
            ):
                raise ValueError('Missing inherited or generated proposition reviews')
        right_required = (
            immutable
            | {
                'spike_clusters.npy',
                'params.py',
                '_phy_cluster_metadata.json',
                '_phy_cluster_metadata_schema.json',
                '_phy_curation_lineage.json',
                '_phy_curation_journal.json',
            }
            | set(right_schema)
        )
        if (source.path / 'curation.json').exists():
            right_required.add('curation_review.json')
        left_required = immutable | {'spike_clusters.npy', 'params.py'} | native_tables
        derived_static = []
        inverse_name = 'whitening_mat_inv.npy'
        if inverse_name not in source.arrays and (left / inverse_name).exists():
            matrix = source.arrays.get('whitening_mat.npy')
            matrix = np.eye(len(source.arrays['channel_map.npy'])) if matrix is None else matrix
            expected_inverse = np.linalg.inv(matrix)
            actual_inverse = np.load(left / inverse_name, allow_pickle=False)
            if actual_inverse.dtype != expected_inverse.dtype or not np.array_equal(
                actual_inverse, expected_inverse
            ):
                raise ValueError('Invalid native derived whitening inverse')
            left_required.add(inverse_name)
            derived_static.append(inverse_name)
        inventories = {}
        for role, path, required in (
            ('ordinary', left, left_required),
            ('replay', right, right_required),
        ):
            names = {p.name for p in path.iterdir()}
            optional = {'phy.log', 'cluster_info.tsv', '.phy'} if role == 'ordinary' else set()
            if role == 'ordinary':
                optional.add('curation_review.json')
                for name in (
                    '_phy_cluster_metadata.json',
                    '_phy_cluster_metadata_schema.json',
                    '_phy_curation_lineage.json',
                    '_phy_curation_journal.json',
                ):
                    if (source.path / name).exists() and (path / name).exists():
                        if _hash(source.path / name) != _hash(path / name):
                            raise ValueError(f'Native source snapshot changed: {name}')
                        optional.add(name)
            if required - names:
                raise ValueError(f'Missing {role} assets: {sorted(required - names)}')
            if names - required - optional:
                raise ValueError(f'Unexpected {role} files: {sorted(names - required - optional)}')
            for name in names:
                item = path / name
                if item.is_symlink() or (item.is_dir() and name != '.phy'):
                    raise ValueError(f'Unexpected non-file output asset: {name}')
            if (path / 'cluster_info.tsv').exists():
                _, summary = _table(path / 'cluster_info.tsv')
                summary_ids = set(next(iter(summary.values())))
                if summary_ids != set(mapped):
                    raise ValueError('Generated cluster_info catalog disagrees')
                for field in {'group', 'notes'} | {
                    f for values in expected_labels.values() for f in values
                }:
                    expected_map = (
                        native.get(field, {})
                        if role == 'ordinary'
                        else replay_metadata.get(field, {})
                    )
                    for unit, value in summary.get(field, {}).items():
                        if unit not in mapped or value != expected_map.get(unit, ''):
                            raise ValueError('Generated cluster_info manual labels disagree')
            inventories[role] = {
                'required': sorted(required),
                'state_artifacts': sorted(names - required),
                'sha256': {name: _hash(path / name) for name in sorted(required)},
            }
        return (
            {
                'ok': True,
                'assignment_oracle': assignment_oracle,
                'immutable_files': sorted(immutable),
                'native_derived_static_files': derived_static,
                'all_metadata_fields': sorted(fields),
                'permitted_metadata_differences': differences,
                'ordinary_aliases': aliases,
                'inventories': inventories,
                'preparation_verified': manifest is not None,
                'saved_generation_verified': saved_generation,
                'historical_parents': (
                    'recorded cumulative ancestry; current parent bound to supplied manifest'
                    if manifest is not None
                    else 'current parent matches original source; preparation not supplied'
                ),
            },
            native,
            replay_metadata,
        )
    except (TypeError, AttributeError, IndexError) as error:
        raise ValueError(f'Malformed contract document: {error}') from error
    finally:
        if curated is not None:
            curated.close()
        if source is not None:
            source.close()
