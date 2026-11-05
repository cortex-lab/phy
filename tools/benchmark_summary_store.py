"""Compare exact unit-indexed summaries with a validated full-source scan.

Run on a fresh COPY of a bounded package: this writes its disposable caches and
store reference. Does not save curation or alter original source arrays. Builds
run in a separate process so their peak RSS is measured independently. No cache
flushing is attempted; timings describe observed, uncontrolled OS cache state.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from phy.apps.large_recording.curation import BoundedSession, project_assignments
from phy.apps.large_recording.live_summaries import LiveSummaryEngine, membership_expressions
from phy.apps.large_recording.source import hash_file
from phy.apps.large_recording.summary_store import attach_summary_store, build_summary_store


def peak_rss():
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    except ImportError:
        return None


class FullScanEngine(LiveSummaryEngine):
    def _indexed(self, expressions, cancel, progress):
        return None


def verify_replay(engine, events, expected):
    """Independent cumulative assignment replay, without expression masks."""
    totals = {unit: engine._empty() for unit in expected}
    required = {e['predicate']['attribute'] for e in events if e['operation'] == 'threshold_split'}
    for start in range(0, engine.manifest['num_spikes'], 1000000):
        stop = start + 1000000
        times = engine.arrays['spike_times.npy'][start:stop].reshape(-1)
        attributes = {name: engine.arrays[name][start:stop].reshape(-1) for name in required}
        assignments = project_assignments(
            engine.arrays['spike_clusters.npy'][start:stop], events, attributes
        )
        scalars = {}
        for field in engine.fields:
            if field == 'amplitude':
                scalars[field] = engine.arrays['amplitudes.npy'][start:stop].reshape(-1)
            elif 'spike_depths.npy' in engine.arrays:
                scalars[field] = engine.arrays['spike_depths.npy'][start:stop].reshape(-1)
            else:
                axis = engine.manifest['source']['identity']['acquisition']['depth_axis']
                scalars[field] = engine.arrays['spike_positions.npy'][start:stop, axis]
        for unit, summary in totals.items():
            selected = assignments == unit
            summary['counts'] += np.histogram(times[selected], bins=engine.edges)[0]
            for field, values in scalars.items():
                finite = selected & np.isfinite(values)
                summary['trend_counts'][field] += np.histogram(times[finite], bins=engine.edges)[0]
                summary['trend_sums'][field] += np.histogram(
                    times[finite], bins=engine.edges, weights=values[finite].astype(np.float64)
                )[0]
    engine._unchanged()
    for unit in expected:
        for name, actual in engine._flat(totals[unit]).items():
            target = engine._flat(expected[unit])[name]
            if np.issubdtype(actual.dtype, np.integer):
                np.testing.assert_array_equal(actual, target)
            else:
                np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-6)


def run(args):
    if args.build_only:
        started = perf_counter()
        build_summary_store(args.bounded, args.store)
        result = {
            'seconds': perf_counter() - started,
            'peak_rss_bytes': peak_rss(),
            'store_bytes': sum(p.stat().st_size for p in args.store.iterdir()),
        }
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        return
    before = {p.name: hash_file(p) for p in args.bounded.iterdir() if p.is_file()}
    build_report = args.output.with_suffix('.build.json')
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
                str(build_report),
                '--build-only',
            ],
            check=True,
        )
    attach_summary_store(args.bounded, args.store)
    session = BoundedSession(args.bounded)
    try:
        events = session.events
        active = membership_expressions(session.unit_ids, events)
        if args.unit not in active:
            raise ValueError(f'Unit {args.unit} is not active.')
        next_id = max(active) + 100
        operation = {
            'operation': 'threshold_split',
            'parent': args.unit,
            'children': [next_id, next_id + 1],
            'predicate': {
                'attribute': 'amplitudes.npy',
                'operator': '<',
                'threshold': args.threshold,
                'nonfinite': 'keep',
            },
        }
        expressions = membership_expressions(session.unit_ids, events + [operation])
        children = {u: expressions[u] for u in operation['children']}
        results, timings, phases = {}, {}, {}
        # Suppress result cache publication for fair repeated uncached jobs.
        # Inputs and caches from existing curation are not modified.
        for mode, engine_class in [('indexed', LiveSummaryEngine), ('full_scan', FullScanEngine)]:
            with engine_class(args.bounded) as engine:
                engine._cached = lambda key: None
                engine._publish = lambda *a: None
                phases[mode] = []

                def progress(phase, done, total, mode=mode):
                    if not phases[mode] or phases[mode][-1][0] != phase:
                        phases[mode].append([phase, done, total])
                    else:
                        phases[mode][-1] = [phase, done, total]

                started = perf_counter()
                results[mode] = engine.resolve(children, progress=progress)
                timings[mode] = perf_counter() - started
                if mode == 'indexed' and not any(
                    p[0] == 'reading prepared unit summaries' for p in phases[mode]
                ):
                    raise ValueError('Indexed benchmark did not use the acceleration store.')
                if mode == 'full_scan':
                    verify_replay(engine, events + [operation], results['indexed'])
                started = perf_counter()
                engine.resolve(children)
                timings[mode + '_repeat'] = perf_counter() - started
        with LiveSummaryEngine(args.bounded) as engine:
            for unit in children:
                expected = engine._flat(results['full_scan'][unit])
                actual = engine._flat(results['indexed'][unit])
                for name in actual:
                    if np.issubdtype(actual[name].dtype, np.integer):
                        np.testing.assert_array_equal(actual[name], expected[name])
                    else:
                        np.testing.assert_allclose(
                            actual[name], expected[name], rtol=1e-9, atol=1e-6
                        )
        after = {p.name: hash_file(p) for p in args.bounded.iterdir() if p.is_file()}
        if before != after:
            raise ValueError('Bounded package files changed.')
        report = {
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'dataset': args.dataset_label,
            'source_spikes': session.manifest['num_spikes'],
            'unit': args.unit,
            'threshold': args.threshold,
            'timings_seconds': timings,
            'phases': phases,
            'counts': {str(u): int(v['counts'].sum()) for u, v in results['indexed'].items()},
            'indexed_equals_full_scan': True,
            'independent_replay_verified_rows': session.manifest['num_spikes'],
            'bounded_files_unchanged': True,
            'cache_conditions': 'Uncontrolled OS cache; source hashing warms the full scan; index built before measurement.',
            'build': json.loads(build_report.read_text()) if build_report.exists() else None,
        }
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bounded', required=True, type=Path)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--unit', type=int, default=49)
    parser.add_argument('--threshold', type=float, default=-341.797119140625)
    parser.add_argument('--dataset-label', default='Unspecified engineering dataset')
    parser.add_argument('--build-only', action='store_true', help=argparse.SUPPRESS)
    run(parser.parse_args())
