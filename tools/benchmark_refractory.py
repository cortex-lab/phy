"""Measure exact RP paths on a disposable bounded-package COPY.

Run each mode in a fresh process. Fresh-process timings include authentication,
with uncontrolled OS cache state; repeated queries measure the in-process cache.
Builders attach disposable artifacts but never save curation. Source mode bypasses
all prepared RP artifacts. This tool never drops OS caches.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from phy.apps.large_recording.live_summaries import LiveSummaryEngine
from phy.apps.large_recording.refractory_engine import RefractoryEngine
from phy.apps.large_recording.refractory_store import build_refractory_summaries
from phy.apps.large_recording.summary_store import SummaryStore, build_summary_store


def peak_rss():
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def fingerprint(result):
    digest = hashlib.sha256()
    for key, value in sorted(result.items()):
        digest.update(key.encode())
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode())
            digest.update(value.tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def run(args):
    report = {'mode': args.mode, 'os_cache': 'uncontrolled', 'queries': []}
    hashed = [0]

    def progress(phase, done, total):
        if phase == 'validating source':
            hashed[0] = max(hashed[0], done)

    started = perf_counter()
    if args.mode == 'build-counters':
        output = build_refractory_summaries(
            args.package, source_dir=args.source_dir, progress=progress
        )
    elif args.mode == 'build-timestamps':
        output = build_summary_store(
            args.package,
            args.store,
            source_dir=args.source_dir,
            timestamps=True,
            progress=progress,
        )
    else:
        output = None
        with LiveSummaryEngine(args.package, source_dir=args.source_dir) as source:
            if args.mode == 'timestamps':
                from phy.apps.large_recording import refractory_store

                refractory_store._counters = lambda *a, **kw: {}
            engine = RefractoryEngine(source, root_bytes=0)
            units = args.units
            original = {'kind': 'source', 'unit': units[0]}
            merged = {'kind': 'union', 'parts': [{'kind': 'source', 'unit': u} for u in units]}
            split = {
                'kind': 'split',
                'parent': merged,
                'selected': True,
                'predicate': {
                    'attribute': 'amplitudes.npy',
                    'operator': '>=',
                    'threshold': args.threshold,
                    'nonfinite': 'keep',
                },
            }
            queries = [('original', original)]
            if args.mode != 'counters':
                queries += [('merge', merged), ('amplitude-child', split)]
            if args.membership != 'all':
                queries = [(label, e) for label, e in queries if label == args.membership]
            if args.mode != 'source':

                def no_source(*a, **kw):
                    raise AssertionError('Prepared benchmark accessed original source.')

                source._open = source._unchanged = no_source
            read_bytes = [0]
            batches = SummaryStore.batches

            def measured_batches(self, *a, **kw):
                for rows in batches(self, *a, **kw):
                    read_bytes[0] += rows.nbytes
                    yield rows

            SummaryStore.batches = measured_batches
            for label, expression in queries:
                for repetition in range(2):
                    before = perf_counter()
                    read_bytes[0] = 0
                    resolve = engine._resolve_source if args.mode == 'source' else engine.resolve
                    value = resolve({0: expression}, progress=progress)[0]
                    report['queries'].append(
                        {
                            'membership': label,
                            'repeat': repetition,
                            'seconds': perf_counter() - before,
                            'path': engine.last_path,
                            'timestamp_bytes': read_bytes[0],
                            'fingerprint': fingerprint(value),
                            'status': value['status'],
                        }
                    )
            if args.mode == 'timestamps':
                cancelled = [False]
                triggered = [None]

                def cancel_after_batch(*args):
                    cancelled[0] = True
                    triggered[0] = perf_counter()

                before = perf_counter()
                # A fresh result engine forces reading; cancel after its first batch.
                try:
                    RefractoryEngine(source).resolve(
                        {0: merged},
                        cancel=lambda: cancelled[0],
                        progress=cancel_after_batch,
                    )
                except Exception as error:
                    from concurrent.futures import CancelledError

                    if not isinstance(error, CancelledError):
                        raise
                    finished = perf_counter()
                    report['cancel_seconds'] = finished - triggered[0]
                    report['cancelled_request_seconds'] = finished - before
    report.update(
        seconds=perf_counter() - started, peak_rss_bytes=peak_rss(), source_hash_bytes=hashed[0]
    )
    if output:
        report['artifact_bytes'] = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'mode', choices=['build-counters', 'build-timestamps', 'source', 'counters', 'timestamps']
    )
    parser.add_argument('package', type=Path)
    parser.add_argument('--store', type=Path)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument(
        '--membership', choices=['all', 'original', 'merge', 'amplitude-child'], default='all'
    )
    parser.add_argument('--units', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
