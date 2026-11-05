"""Independent acceptance comparison for complete ordinary and bounded outputs.

Manual protocol (same ledger on both branches): alias each original unit ID,
merge parents into ``as``, split all spikes of ``parent`` with finite amplitude
strictly below ``threshold`` into ``selected`` (equal/nonfinite => ``remainder``),
then explicitly label every final unit's group and notes. Undo/redo apply to
curation commands; save/reopen are history boundaries. Save, reopen, save again
without editing, and preserve both outputs. Compare COMPLETE replay arrays,
never bounded evidence arrays. Producer metrics (SNR etc.) are excluded: clearing
them on transformed units is an intentional bounded metadata policy.

Ledger JSON: {"aliases": {"a": 0, "b": 1}, "actions": [
 {"op": "merge", "parents": ["a", "b"], "as": "ab"},
 {"op": "split", "parent": "ab", "threshold": 1.5,
  "selected": "low", "remainder": "rest"},
 {"op": "label", "ids": ["low"], "field": "group", "value": "good"},
 {"op": "undo"}, {"op": "redo"}, {"op": "save"}, {"op": "reopen"}],
 "expected_labels": {"low": {"group": "good"}}, "fields": ["group", "notes"]}.
Explicit final labels avoid comparing legacy ordinary metadata inheritance with
bounded transformed-unit clearing. Use each GUI's reported IDs to track aliases.
Amplitude predicates must use raw amplitudes.npy; TemplateModel sanitizes nonfinite
values to zero, so GUI lasso/predicate selection on displayed amplitudes alone is
not this protocol. The ordinary engine runs full Supervisor commands without attaching GUI views;
the streamed engine is a separate baseline, NOT full ordinary GUI evidence.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def labels(path, fields):
    result = {field: {} for field in fields}
    tables = sorted(Path(path).glob('cluster_*.tsv')) + sorted(Path(path).glob('cluster_*.csv'))
    for table in tables:
        # GUI-generated summary duplicates authoritative label tables.
        if table.stem == 'cluster_info':
            continue
        with table.open(newline='') as stream:
            reader = csv.DictReader(stream, delimiter=',' if table.suffix == '.csv' else '\t')
            if 'cluster_id' not in (reader.fieldnames or []):
                raise ValueError(f'Label table lacks cluster_id: {table}')
            for row in reader:
                unit = int(row['cluster_id'])
                for field in fields:
                    if field in row:
                        if unit in result[field]:
                            raise ValueError(f'Duplicate {field} label for {unit}')
                        result[field][unit] = row[field]
    return result


def write_labels(path, data):
    for field, values in data.items():
        if not field.isidentifier():
            raise ValueError('Label fields must be identifiers')
        with (Path(path) / f'cluster_{field}.tsv').open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['cluster_id', field])
            writer.writerows(sorted(values.items()))


def compare_outputs(
    left,
    right,
    fields=('group', 'notes'),
    chunk_size=100000,
    source=None,
    expected_labels=None,
    bounded_dir=None,
):
    """Exact all-row partition equivalence, O(chunk_size + number of units) RAM.

    A two-way raw-ID mapping proves equivalence without partition hash collisions.
    SHA256 is used only to identify saved inputs/coordinates, not as that proof.
    """
    report = {
        'ok': False,
        'method': 'exact all-row two-way ID mapping',
        'revision': revision(),
        'left': str(Path(left).resolve()),
        'right': str(Path(right).resolve()),
    }
    try:
        if chunk_size <= 0:
            raise ValueError('chunk_size must be positive')
        arrays = [
            np.load(Path(p) / 'spike_clusters.npy', mmap_mode='r', allow_pickle=False)
            for p in (left, right)
        ]
        times = [
            np.load(Path(p) / 'spike_times.npy', mmap_mode='r', allow_pickle=False)
            for p in (left, right)
        ]
        for a in arrays + times:
            if a.ndim not in (1, 2) or (a.ndim == 2 and a.shape[1] != 1):
                raise ValueError('Expected scalar per-spike arrays')
            if a.dtype.kind not in 'iu':
                raise ValueError('Assignments and source sample times must be integers, no NaNs')
        arrays = [a.reshape(-1) for a in arrays]
        times = [a.reshape(-1) for a in times]
        if len({len(a) for a in arrays + times}) != 1:
            raise ValueError('Unequal source/output row counts')
        report['rows'] = len(arrays[0])
        report['assignment_dtypes'] = [str(a.dtype) for a in arrays]
        if times[0].dtype != times[1].dtype:
            raise ValueError('Unequal source time dtypes')
        forward, reverse = {}, {}
        mismatches, examples = 0, []
        for start in range(0, len(arrays[0]), chunk_size):
            stop = start + chunk_size
            if not np.array_equal(times[0][start:stop], times[1][start:stop]):
                raise ValueError(f'Source time row alignment differs at chunk {start}')
            aa, bb = arrays[0][start:stop], arrays[1][start:stop]
            if np.any(aa < 0) or np.any(bb < 0):
                raise ValueError('Negative cluster ID')
            # Unique raw pairs retain exact integer comparison; process first occurrences
            # in source row order so diagnostics are stable across chunk sizes.
            pairs = np.empty(len(aa), dtype=[('left', aa.dtype), ('right', bb.dtype)])
            pairs['left'], pairs['right'] = aa, bb
            unique, first, counts = np.unique(pairs, return_index=True, return_counts=True)
            for pair_index in np.argsort(first):
                a, b = int(unique[pair_index]['left']), int(unique[pair_index]['right'])
                if forward.get(a, b) != b or reverse.get(b, a) != a:
                    mismatches += int(counts[pair_index])
                    if len(examples) < 10:
                        examples.append(
                            {'row': start + int(first[pair_index]), 'left': a, 'right': b}
                        )
                else:
                    forward[a], reverse[b] = b, a
        report.update(
            partition_mismatch_rows=mismatches,
            examples=examples,
            id_mapping={str(k): v for k, v in sorted(forward.items())},
        )
        report['saved_provenance'] = {
            side: {
                name: json.loads((Path(p) / name).read_text())
                for name in ('comparison_reference.json', '_phy_curation_lineage.json')
                if (Path(p) / name).exists()
            }
            for side, p in (('left', left), ('right', right))
        }
        report['input_hashes'] = {
            side: {
                name: digest(Path(p) / name) for name in ('spike_times.npy', 'spike_clusters.npy')
            }
            for side, p in (('left', left), ('right', right))
        }
        permitted = set()
        if source is not None:
            if __package__:
                from tools.curation_output_contract import compare_file_contract
            else:
                from curation_output_contract import compare_file_contract

            contract, a_labels, b_labels = compare_file_contract(
                left, right, source, forward, expected_labels, bounded_dir, chunk_size
            )
            report['file_contract'] = contract
            permitted = {
                (d['field'], d['left_id'], d['right_id'], d['left'], d['right'])
                for d in contract['permitted_metadata_differences']
            }
            report['permitted_membership_policy_differences'] = contract[
                'permitted_metadata_differences'
            ]
        else:
            a_labels, b_labels = labels(left, fields), labels(right, fields)
        differences = []
        for field in fields:
            for a, b in forward.items():
                av, bv = a_labels.get(field, {}).get(a, ''), b_labels.get(field, {}).get(b, '')
                if av != bv and (field, a, b, av, bv) not in permitted:
                    differences.append(
                        {'field': field, 'left_id': a, 'right_id': b, 'left': av, 'right': bv}
                    )
        unmapped = {
            'left': {
                field: sorted(set(values) - set(forward)) for field, values in a_labels.items()
            },
            'right': {
                field: sorted(set(values) - set(reverse)) for field, values in b_labels.items()
            },
        }
        report['unmapped_label_ids'] = unmapped
        report['manual_label_mismatches'] = differences
        report['compared_fields'] = list(fields)
        report['ok'] = (
            mismatches == 0
            and not differences
            and not any(ids for side in unmapped.values() for ids in side.values())
        )
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError) as error:
        report['error'] = str(error)
    return report


def run_reference(source, ledger, output, engine='ordinary', chunk_size=100000):
    """Execute a common action ledger independently of bounded journal code.

    Streaming uses disk snapshots and direct masks. Ordinary uses full Supervisor
    split with ALL matching source row IDs, never a bounded predicate projector.
    Output is a comparison artifact (assignments, times, manual TSVs), not a full
    sorting export. No source file is modified.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError('Output must be new and outside the immutable source')
    if engine not in ('ordinary', 'streamed') or chunk_size <= 0:
        raise ValueError('Invalid reference engine or chunk size')
    fields = ledger.get('fields', ['group', 'notes'])
    data = labels(source, fields)
    aliases = {k: int(v) for k, v in ledger['aliases'].items()}
    amplitudes = np.load(source / 'amplitudes.npy', mmap_mode='r').reshape(-1)
    output.mkdir(parents=True)
    shutil.copyfile(source / 'spike_times.npy', output / 'spike_times.npy')
    shutil.copyfile(source / 'spike_clusters.npy', output / 'spike_clusters.npy')
    history, future = [], []
    snapshots = []
    state = None
    current = None
    allocator = 0
    callback_ids = set()

    def dispose():
        nonlocal state, current
        from phylib.utils.event import _EVENT

        _EVENT._callbacks = [item for item in _EVENT._callbacks if id(item) not in callback_ids]
        callback_ids.clear()
        if engine == 'streamed' and state is not None:
            state.flush()
            state._mmap.close()
        current = None
        state = None
        # Supervisor contains cycles through history and action creators.
        gc.collect()

    def reopen():
        nonlocal state, allocator
        dispose()
        if engine == 'ordinary':
            from phy.cluster.supervisor import Supervisor

            class ReferenceSupervisor(Supervisor):
                # Presentation hooks only: native command/history/metadata code is unchanged.
                def _apply_selection_change(self, *args, **kwargs):
                    pass

                def _set_merge_mode_ui(self, *args):
                    pass

                def _reveal_cluster_result(self, *args):
                    pass

                def _hide_merge_view(self):
                    pass

                def _workflow_context(self):
                    return {}

                def _reset_cluster_view(self):
                    pass

            from phylib.utils.event import _EVENT

            existing = {id(item) for item in _EVENT._callbacks}
            try:
                state = ReferenceSupervisor(
                    np.load(output / 'spike_clusters.npy').reshape(-1),
                    cluster_groups=data.get('group', {}),
                    cluster_labels={f: d for f, d in data.items() if f != 'group'},
                    new_cluster_id=allocator or None,
                )
            finally:
                callback_ids.update(
                    id(item) for item in _EVENT._callbacks if id(item) not in existing
                )
            state.cluster_view = state.similarity_view = state.select_actions = None
        else:
            state = np.load(output / 'spike_clusters.npy', mmap_mode='r+').reshape(-1)

    def assignments():
        return state.clustering.spike_clusters if engine == 'ordinary' else state

    def persist():
        nonlocal data
        if engine == 'ordinary':
            data = {field: state.get_labels(field) for field in fields}
            np.save(output / 'spike_clusters.npy', assignments())
        else:
            state.flush()
        write_labels(output, data)

    try:
        reopen()
        for index, action in enumerate(ledger['actions']):
            op = action['op']
            if op in ('save', 'reopen'):
                persist()
                if op == 'reopen':
                    reopen()
                    history.clear()
                    future.clear()
                continue
            if op in ('undo', 'redo'):
                if engine == 'ordinary':
                    getattr(state, op)()
                else:
                    origin, destination = (history, future) if op == 'undo' else (future, history)
                    if not origin:
                        raise ValueError(f'No {op} available')
                    snap = output / f'.history-{index}.npy'
                    state.flush()
                    shutil.copyfile(output / 'spike_clusters.npy', snap)
                    saved_array = np.load(snap, mmap_mode='r').reshape(-1)
                    snapshots.append(saved_array)
                    destination.append((saved_array, copy.deepcopy(data)))
                    saved, data = origin.pop()
                    for start in range(0, len(state), chunk_size):
                        state[start : start + chunk_size] = saved[start : start + chunk_size]
                continue
            if engine == 'streamed':
                # Disk snapshots keep RAM bounded even for large reference runs.
                snap = output / f'.history-{index}.npy'
                state.flush()
                shutil.copyfile(output / 'spike_clusters.npy', snap)
                saved_array = np.load(snap, mmap_mode='r').reshape(-1)
                snapshots.append(saved_array)
                history.append((saved_array, copy.deepcopy(data)))
                future.clear()
            if op == 'label':
                ids = [aliases[a] for a in action['ids']]
                active = set()
                for start in range(0, len(assignments()), chunk_size):
                    active.update(map(int, np.unique(assignments()[start : start + chunk_size])))
                if not ids or not set(ids) <= active:
                    raise ValueError('Label aliases must refer to active units')
                if engine == 'ordinary':
                    state.label(action['field'], action['value'], ids)
                else:
                    data.setdefault(action['field'], {}).update(
                        dict.fromkeys(ids, action['value'])
                    )
                continue
            current = assignments()
            next_id = max(
                allocator,
                (
                    max(
                        (
                            int(np.max(current[start : start + chunk_size]))
                            for start in range(0, len(current), chunk_size)
                        ),
                        default=-1,
                    )
                    + 1
                ),
            )
            if engine == 'ordinary':
                next_id = int(state.clustering.new_cluster_id())
            active = set()
            for start in range(0, len(current), chunk_size):
                active.update(map(int, np.unique(current[start : start + chunk_size])))
            if op == 'merge':
                parents = [aliases[a] for a in action['parents']]
                if (
                    len(parents) < 2
                    or len(set(parents)) != len(parents)
                    or not set(parents) <= active
                ):
                    raise ValueError('Merge requires distinct active parent aliases')
                if engine == 'ordinary':
                    state.merge(parents)
                else:
                    for start in range(0, len(current), chunk_size):
                        block = current[start : start + chunk_size]
                        block[np.isin(block, parents)] = next_id
                if engine == 'streamed':
                    for values in data.values():
                        for parent in parents:
                            values.pop(parent, None)
                allocator = next_id + 1
                aliases[action['as']] = next_id
            elif op == 'split':
                parent, threshold = aliases[action['parent']], float(action['threshold'])
                if parent not in active:
                    raise ValueError('Split parent alias is inactive')
                if not np.isfinite(threshold):
                    raise ValueError('Threshold must be finite')
                if engine == 'ordinary':
                    selected = np.flatnonzero(
                        (current == parent) & np.isfinite(amplitudes) & (amplitudes < threshold)
                    )
                    if not len(selected) or len(selected) == np.count_nonzero(current == parent):
                        raise ValueError('Split must have both selected and remainder rows')
                    state.split(selected)
                else:
                    selected_count, remainder_count = 0, 0
                    for start in range(0, len(current), chunk_size):
                        mask = current[start : start + chunk_size] == parent
                        values = amplitudes[start : start + chunk_size]
                        below = np.isfinite(values) & (values < threshold)
                        selected_count += int(np.count_nonzero(mask & below))
                        remainder_count += int(np.count_nonzero(mask & ~below))
                    if not selected_count or not remainder_count:
                        raise ValueError('Split must have both selected and remainder rows')
                    for start in range(0, len(current), chunk_size):
                        block = current[start : start + chunk_size]
                        values = amplitudes[start : start + chunk_size]
                        mask = block == parent
                        below = np.isfinite(values) & (values < threshold)
                        block[mask & below] = next_id
                        block[mask & ~below] = next_id + 1
                if engine == 'streamed':
                    for values in data.values():
                        values.pop(parent, None)
                allocator = next_id + 2
                aliases[action['selected']], aliases[action['remainder']] = next_id, next_id + 1
            else:
                raise ValueError(f'Unknown ledger operation: {op}')
        persist()
        for alias, expected in ledger.get('expected_labels', {}).items():
            for field, value in expected.items():
                if str(data[field].get(aliases[alias], '')) != str(value):
                    raise ValueError(f'Final label expectation failed: {alias}.{field}')
        report = {
            'engine': engine,
            'evidence': (
                'full ordinary Supervisor, no GUI views attached'
                if engine == 'ordinary'
                else 'streamed reference, NOT full ordinary GUI'
            ),
            'revision': revision(),
            'source': str(source),
            'aliases': aliases,
            'ledger': ledger,
            'input_hashes': {
                name: digest(source / name)
                for name in ('spike_times.npy', 'spike_clusters.npy', 'amplitudes.npy')
            },
        }
        (output / 'comparison_reference.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    finally:
        for saved_array in snapshots:
            saved_array._mmap.close()
        for snap in output.glob('.history-*.npy'):
            snap.unlink()
        dispose()
        mapping = getattr(amplitudes, '_mmap', None)
        if mapping is not None:
            mapping.close()


def report_path(path, inputs):
    """Reject existing paths and all locations inside either immutable input."""
    path = Path(path).resolve()
    for root in inputs:
        root = Path(root).resolve()
        if path == root or root in path.parents:
            raise ValueError('Report must be outside both input directories')
    if path.exists():
        raise FileExistsError(f'Report already exists: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    compare = sub.add_parser('compare')
    compare.add_argument('ordinary')
    compare.add_argument('bounded_replay')
    compare.add_argument('--fields', nargs='+', default=['group', 'notes'])
    compare.add_argument('--output', required=True)
    compare.add_argument('--chunk-size', type=int, default=100000)
    compare.add_argument(
        '--source', help='Original immutable sorting: enable the whole-file contract.'
    )
    compare.add_argument(
        '--expectations', help='JSON map of replay unit IDs to explicit final field values.'
    )
    compare.add_argument(
        '--bounded-dir', help='Final round package, to verify preparation/parent binding.'
    )
    reference = sub.add_parser('reference')
    reference.add_argument('source')
    reference.add_argument('ledger')
    reference.add_argument('output')
    reference.add_argument('--engine', choices=['ordinary', 'streamed'], default='ordinary')
    reference.add_argument('--chunk-size', type=int, default=100000)
    args = parser.parse_args()
    if args.command == 'compare':
        try:
            if (args.expectations or args.bounded_dir) and not args.source:
                raise ValueError('--expectations and --bounded-dir require --source')
            inputs = [args.ordinary, args.bounded_replay]
            if args.bounded_dir:
                inputs.append(args.bounded_dir)
            if args.source:
                source_path = Path(args.source).resolve()
                inputs.append(source_path.parent if source_path.is_file() else source_path)
            destination = report_path(args.output, inputs)
            result = compare_outputs(
                args.ordinary,
                args.bounded_replay,
                args.fields,
                args.chunk_size,
                source=args.source,
                expected_labels=json.loads(Path(args.expectations).read_text())
                if args.expectations
                else None,
                bounded_dir=args.bounded_dir,
            )
            # Exclusive creation also rejects another writer winning after validation.
            with destination.open('x') as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write('\n')
        except (ValueError, OSError) as error:
            print(json.dumps({'ok': False, 'error': str(error)}))
            return 1
        print(json.dumps({'ok': result['ok'], 'report': args.output}))
        return 0 if result['ok'] else 1
    try:
        result = run_reference(
            args.source,
            json.loads(Path(args.ledger).read_text()),
            args.output,
            args.engine,
            args.chunk_size,
        )
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({'ok': False, 'error': str(error)}))
        return 1
    print(json.dumps({'ok': True, 'evidence': result['evidence'], 'output': args.output}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
