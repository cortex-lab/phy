"""Exact live-summary scaling benchmark; original source is read-only.

The bounded package must be a fresh benchmark preparation/copy. Curation is
never saved. Derived result cache use/publication is disabled for all timings.
OS caches are uncontrolled: these are not cold-storage latency guarantees.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from phy.apps.large_recording.curation import BoundedSession, project_assignments
from phy.apps.large_recording.live_summaries import LiveSummaryEngine, membership_expressions
from phy.apps.large_recording.source import hash_file
from phy.apps.large_recording.summary_store import (
    SummaryStore,
    attach_summary_store,
    build_summary_store,
)


def progress_logger():
    last = [None, 0.0]

    def progress(phase, done, total):
        now = perf_counter()
        if phase != last[0] or now - last[1] > 10 or done == total:
            print(f'{phase}: {100 * done / max(1, total):.1f}% ({done:,}/{total:,})', flush=True)
            last[:] = [phase, now]

    return progress


def no_cache(engine):
    engine._cached = lambda key: None
    engine._publish = lambda *args: None
    return engine


def compare(engine, actual, expected):
    for unit in expected:
        for name, values in engine._flat(actual[unit]).items():
            target = engine._flat(expected[unit])[name]
            if np.issubdtype(values.dtype, np.integer):
                np.testing.assert_array_equal(values, target)
            else:
                np.testing.assert_allclose(values, target, rtol=1e-9, atol=1e-6)


def ancestors(e):
    if e['kind'] == 'source':
        return {e['unit']}
    if e['kind'] == 'split':
        return ancestors(e['parent'])
    return set().union(*(ancestors(p) for p in e['parts']))


def split(parent, children, threshold):
    return {
        'operation': 'threshold_split',
        'parent': int(parent),
        'children': children,
        'predicate': {
            'attribute': 'amplitudes.npy',
            'operator': '<',
            'threshold': float(threshold),
            'nonfinite': 'keep',
        },
    }


def run(args):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    from phy.apps.large_recording import curation, live_summaries, source, summary_store

    components = {
        Path(module.__file__).name: hash_file(module.__file__)
        for module in (curation, live_summaries, source, summary_store)
    }
    started = perf_counter()
    if args.build_only:
        build_summary_store(args.bounded, args.store, progress=progress_logger())
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            rss = None
        result = {
            'seconds': perf_counter() - started,
            'peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss * 1024)
            if rss is not None
            else None,
            'store_bytes': sum(p.stat().st_size for p in args.store.iterdir()),
        }
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        return
    if not args.store.exists():
        subprocess.run(
            [
                sys.executable,
                __file__,
                '--bounded',
                str(args.bounded),
                '--store',
                str(args.store),
                '--output',
                str(args.output.with_suffix('.build.json')),
                '--build-only',
            ],
            check=True,
        )
    attach_summary_store(args.bounded, args.store)
    before = {p.name: hash_file(p) for p in args.bounded.iterdir() if p.is_file()}
    session = BoundedSession(args.bounded)
    try:
        base = session.events
        if base:
            raise ValueError('Use a fresh, uncurated bounded package.')
        m = session.manifest
        counts = {int(u): int(n) for u, n in zip(m['unit_ids'], m['counts']) if n > 0}
        ordered = sorted(counts, key=lambda u: counts[u])
        median = ordered[len(ordered) // 2]
        largest = ordered[-1]
        middle = max(0, len(ordered) // 2 - 5)
        typical = ordered[middle : middle + 10]
        thresholds = {}
        with LiveSummaryEngine(args.bounded) as engine, SummaryStore(engine) as store:
            for unit in {median, largest, typical[len(typical) // 2]}:
                sample = []
                total = 0
                for data in store.batches(unit, 1000000):
                    if total < 16384:
                        finite = data['amplitude'][np.isfinite(data['amplitude'])][: 16384 - total]
                        sample.append(finite)
                        total += len(finite)
                values = np.concatenate(sample)
                if len(values) < 2:
                    raise ValueError('No usable amplitude threshold sample.')
                thresholds[unit] = [
                    float(np.quantile(values, 0.25)),
                    float(np.quantile(values, 0.5)),
                ]
        next_id = max(m['unit_ids']) + 100
        recipes = []

        def add(name, events, children):
            expressions = membership_expressions(session.unit_ids, events)
            recipes.append((name, events, {u: expressions[u] for u in children}))

        for name, unit in [('median_unit', median), ('largest_unit', largest)]:
            children = [next_id, next_id + 1]
            next_id += 100
            add(name, [split(unit, children, thresholds[unit][1])], children)
        children = [next_id, next_id + 1]
        grandchildren = [next_id + 2, next_id + 3]
        next_id += 100
        add(
            'nested_split_median',
            [
                split(median, children, thresholds[median][1]),
                split(children[1], grandchildren, thresholds[median][0]),
            ],
            grandchildren,
        )
        for name, parents, threshold in [
            ('merge_two_largest_then_split', ordered[-2:], thresholds[largest][1]),
            ('merge_ten_typical_then_split', typical, thresholds[typical[len(typical) // 2]][1]),
        ]:
            merged = next_id
            children = [next_id + 1, next_id + 2]
            next_id += 100
            add(
                name,
                [
                    {'operation': 'merge', 'parents': list(map(int, parents)), 'target': merged},
                    split(merged, children, threshold),
                ],
                children,
            )
        indexed = {}
        reports = []
        with (
            LiveSummaryEngine(args.bounded) as binding_engine,
            SummaryStore(binding_engine) as store,
        ):
            record_bytes = store.dtype.itemsize
        # Each scenario gets a fresh engine. A second uncached request tests
        # repeat reads, not a result-cache hit. No source fallback is accepted.
        for name, events, expressions in recipes:
            with no_cache(LiveSummaryEngine(args.bounded)) as engine:

                def forbid_source(*a, **kw):
                    raise AssertionError('Indexed request opened original source')

                engine._open = forbid_source
                progress = []
                start = perf_counter()
                result = engine.resolve(
                    expressions, progress=lambda *p, progress=progress: progress.append(p)
                )
                first = perf_counter() - start
                start = perf_counter()
                repeat = engine.resolve(expressions)
                second = perf_counter() - start
                compare(engine, repeat, result)
                indexed.update(result)
                original = set().union(*(ancestors(e) for e in expressions.values()))
                rows = sum(counts[u] for u in original)
                report = {
                    'scenario': name,
                    'events': events,
                    'original_units': sorted(original),
                    'source_rows_read': rows,
                    'payload_bytes_read': rows * record_bytes,
                    'first_seconds': first,
                    'repeat_seconds': second,
                    'counts': {str(u): int(v['counts'].sum()) for u, v in result.items()},
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
        expressions = {u: e for _, _, expr in recipes for u, e in expr.items()}
        # One validated full-source scan computes all scenario expressions.
        with no_cache(LiveSummaryEngine(args.bounded)) as engine:
            engine._indexed = lambda *a: None
            start = perf_counter()
            baseline = engine.resolve(expressions, progress=progress_logger())
            full_seconds = perf_counter() - start
            compare(engine, indexed, baseline)
            # A separate sequential semantic replay is independent of the
            # expression-mask resolver and traverses every original source row.
            oracle = {u: engine._empty() for u in expressions}
            total = m['num_spikes']
            progress = progress_logger()
            start = perf_counter()
            for first in range(0, total, 1000000):
                stop = min(total, first + 1000000)
                times = engine.arrays['spike_times.npy'][first:stop].reshape(-1)
                assignments = engine.arrays['spike_clusters.npy'][first:stop].reshape(-1)
                amplitude = engine.arrays['amplitudes.npy'][first:stop].reshape(-1)
                scalars = {'amplitude': amplitude}
                if 'depth' in engine.fields:
                    if 'spike_depths.npy' in engine.arrays:
                        scalars['depth'] = engine.arrays['spike_depths.npy'][first:stop].reshape(
                            -1
                        )
                    else:
                        axis = m['source']['identity']['acquisition']['depth_axis']
                        scalars['depth'] = engine.arrays['spike_positions.npy'][first:stop, axis]
                for name, events, targets in recipes:
                    projected = project_assignments(
                        assignments, events, {'amplitudes.npy': amplitude}
                    )
                    for unit in targets:
                        selected = projected == unit
                        oracle[unit]['counts'] += np.histogram(times[selected], bins=engine.edges)[
                            0
                        ]
                        for field, values in scalars.items():
                            finite = selected & np.isfinite(values)
                            oracle[unit]['trend_counts'][field] += np.histogram(
                                times[finite], bins=engine.edges
                            )[0]
                            oracle[unit]['trend_sums'][field] += np.histogram(
                                times[finite],
                                bins=engine.edges,
                                weights=values[finite].astype(np.float64),
                            )[0]
                progress('independent semantic replay', stop, total)
            oracle_seconds = perf_counter() - start
            engine._unchanged()
            compare(engine, indexed, oracle)
        after = {p.name: hash_file(p) for p in args.bounded.iterdir() if p.is_file()}
        if before != after:
            raise ValueError('Package files changed')
        build_report = args.output.with_suffix('.build.json')
        report = {
            'revision': revision,
            'component_sha256': components,
            'label': args.label,
            'source_spikes': m['num_spikes'],
            'duration_hours': m['duration_seconds'] / 3600,
            'native_bins': len(engine.edges) - 1,
            'source_units': len(m['unit_ids']),
            'source_dir': m['source']['path'],
            'cache_conditions': 'Uncontrolled OS page cache; no summary result-cache hits or publication. Indexed timings exclude GUI delivery.',
            'hardware': {
                'platform': platform.platform(),
                'ram_bytes': os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
                if hasattr(os, 'sysconf')
                else None,
            },
            'build': json.loads(build_report.read_text()) if build_report.exists() else None,
            'build_note': 'Build cost may be from an earlier invocation; runtime component hashes identify this verification run.',
            'scenarios': reports,
            'combined_full_source_scan_seconds': full_seconds,
            'independent_replay_seconds': oracle_seconds,
            'independent_replay_rows': m['num_spikes'],
            'indexed_equals_full_scan_and_independent_replay': True,
            'bounded_files_unchanged': True,
        }
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('DONE', args.output, flush=True)
    finally:
        session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bounded', required=True, type=Path)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--label', default='Unspecified engineering dataset')
    parser.add_argument('--build-only', action='store_true')
    run(parser.parse_args())
