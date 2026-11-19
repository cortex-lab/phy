"""Profile bounded commands on a new synthetic sorting, never a supplied dataset.

Separate setup from action latency. OS caches are uncontrolled; optional cProfile
results are diagnostic and are not mixed into the unprofiled timing samples.
"""

import argparse
import cProfile
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phy.apps.large_recording import curation
from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.supervisor import BoundedSupervisor
from tools.benchmark_long_recording import _make_source


def _journal(units, events, merges):
    """A fixed merge prefix followed by metadata-only work keeps the live catalog fixed."""
    result = []
    target = 0
    for index in range(merges):
        next_id = units + index
        result.append(
            {
                'version': 2,
                'operation': 'merge',
                'parents': [target, index + 1],
                'target': next_id,
                'metadata': {'group': None},
            }
        )
        target = next_id
    while len(result) < events:
        result.append(
            {
                'version': 2,
                'operation': 'label',
                'field': 'group',
                'ids': [units - 1],
                'value': 'good' if len(result) % 2 else 'mua',
            }
        )
    return result


def run(args):
    if args.spikes < args.units or args.units < 3 or args.repeats < 1:
        raise ValueError('Require spikes >= units >= 3 and repeats >= 1.')
    if not 0 <= args.merges < args.units - 1:
        raise ValueError('Merge prefix must leave the measured original unit active.')
    if any(events < args.merges for events in args.events):
        raise ValueError('Every journal length must include the fixed merge prefix.')
    args.output.mkdir(parents=True, exist_ok=False)
    setup = perf_counter()
    params = args.output / 'source' / 'params.py'
    _make_source(params.parent, args.spikes, args.units, 1000, args.batch_size)
    bounded = prepare_large(
        params,
        args.output / 'bounded',
        window_seconds=3_600_000,
        stratum_seconds=3_600_000,
        max_spikes=args.spikes,
        bin_seconds=1800,
        batch_size=args.batch_size,
    )
    setup_seconds = perf_counter() - setup
    original = curation.project_assignments
    calls = []

    def measured_projection(assignments, *a, **kw):
        started = perf_counter()
        result = original(assignments, *a, **kw)
        calls.append({'rows': len(assignments), 'seconds': perf_counter() - started})
        return result

    report = {
        'revision': subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        'python': sys.version,
        'numpy': np.__version__,
        'platform': platform.platform(),
        'source_spikes': args.spikes,
        'units': args.units,
        'fixed_merges': args.merges,
        'setup_seconds': setup_seconds,
        'cache_conditions': 'uncontrolled OS caches; warm synthetic arrays',
        'measurements': [],
    }
    try:
        for length in args.events:
            with curation.BoundedSession(bounded) as session:
                events = _journal(args.units, length, args.merges)
                _, _, next_id = curation.interpret_events(
                    session.unit_ids, session.base_metadata, events
                )
                session.restore_journal({'events': tuple(events), 'next_cluster_id': next_id})
                supervisor = BoundedSupervisor(
                    spike_clusters=session.assignments.copy(),
                    session=session,
                    cluster_groups=session.metadata.get('group', {}),
                    cluster_labels=session.metadata,
                )
                samples = []
                for repeat in range(args.repeats):
                    calls.clear()
                    curation.project_assignments = measured_projection
                    wall, cpu = perf_counter(), process_time()
                    try:
                        supervisor.label(
                            'group', 'good' if repeat % 2 else 'mua', [args.units - 1]
                        )
                    finally:
                        curation.project_assignments = original
                    samples.append(
                        {
                            'wall_seconds': perf_counter() - wall,
                            'cpu_seconds': process_time() - cpu,
                            'projector_calls': list(calls),
                        }
                    )
                if args.profile:
                    profile = cProfile.Profile()
                    profile.runcall(supervisor.label, 'group', 'good', [args.units - 1])
                    profile.dump_stats(args.output / f'label-{length}.prof')
                actual = session.assignments
                expected = original(session.base_assignments, session.events, session.attributes)
                np.testing.assert_array_equal(actual, expected)
                row = {
                    'initial_events': length,
                    'samples': samples,
                    'p50_seconds': float(np.median([s['wall_seconds'] for s in samples])),
                    'p95_seconds': float(np.percentile([s['wall_seconds'] for s in samples], 95)),
                }
                report['measurements'].append(row)
                print(json.dumps(row), flush=True)
                del supervisor
    finally:
        curation.project_assignments = original
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == 'darwin' else 1024
        )
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spikes', type=int, default=1_000_000)
    parser.add_argument('--units', type=int, default=859)
    parser.add_argument('--merges', type=int, default=100)
    parser.add_argument('--events', type=int, nargs='+', default=[100, 213, 1000])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=100_000)
    parser.add_argument('--profile', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
