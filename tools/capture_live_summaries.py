"""Capture and verify a live full-source summary recomputation on a package copy.

Example (requires a display or xvfb-run)::

    uv run --no-sync python tools/capture_live_summaries.py \
        --bounded /path/to/bounded --output /fresh/capture --unit 49 --control 50

Inputs are read only. The output retains working-bounded/, config/, three real
GUI screenshots and capture.json. Hash preflight warms the filesystem cache;
reported first-job latency includes worker validation and scan, but is not a
cold-storage benchmark. This tool does not establish acquisition exposure truth.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from capture_long_recording_views import _capture, _close_capture

from phy.apps.large_recording.curation import project_assignments
from phy.apps.large_recording.source import close_array, hash_file, hash_json
from phy.apps.template.gui import TemplateController
from phy.gui.qt import QApplication, Qt

REPOSITORY = Path(__file__).resolve().parents[1]
PROVENANCE = 'Converted engineering validation; acquisition exposure unverified'


def _package_hashes(path):
    return {str(p.relative_to(path)): hash_file(p) for p in sorted(path.rglob('*')) if p.is_file()}


def _required_arrays(manifest, events):
    names = {'spike_times.npy', 'spike_clusters.npy', 'amplitudes.npy'}
    names.update(
        event['predicate']['attribute']
        for event in events
        if event['operation'] == 'threshold_split'
    )
    if 'depth_sum_file' in manifest['summaries']:
        names.add(
            'spike_depths.npy'
            if 'spike_depths.npy' in manifest['source_array_inventory']
            else 'spike_positions.npy'
        )
    return sorted(names)


def _wait(predicate, timeout, description, service):
    deadline = time.perf_counter() + timeout
    while not predicate():
        QApplication.processEvents()
        if service._errors:
            raise RuntimeError(f'{description}: {list(service._errors.values())}')
        if time.perf_counter() >= deadline:
            raise TimeoutError(description)
        time.sleep(0.01)


def _verify_source(source, manifest, events, actual, batch_size):
    """Replay every source row independently of the membership-expression engine."""
    arrays = {}
    started = time.perf_counter()
    try:
        for name in _required_arrays(manifest, events):
            arrays[name] = np.load(source / name, mmap_mode='r', allow_pickle=False)
        edges = np.asarray(actual[next(iter(actual))]['edges'])
        bins = len(edges) - 1
        fields = list(actual[next(iter(actual))]['trend_sums'])
        expected = {
            unit: {
                'counts': np.zeros(bins, dtype=np.int64),
                'trend_sums': {f: np.zeros(bins, dtype=np.float64) for f in fields},
                'trend_counts': {f: np.zeros(bins, dtype=np.int64) for f in fields},
            }
            for unit in actual
        }
        n_spikes = manifest['num_spikes']
        for start in range(0, n_spikes, batch_size):
            stop = min(start + batch_size, n_spikes)
            batch = {
                name: array[start:stop].reshape(-1)
                for name, array in arrays.items()
                if name != 'spike_positions.npy'
            }
            assignments = project_assignments(batch['spike_clusters.npy'], events, batch)
            times = batch['spike_times.npy']
            scalars = {'amplitude': batch['amplitudes.npy']}
            if 'depth' in fields:
                if 'spike_depths.npy' in batch:
                    scalars['depth'] = batch['spike_depths.npy']
                else:
                    axis = manifest['source']['identity']['acquisition']['depth_axis']
                    scalars['depth'] = arrays['spike_positions.npy'][start:stop, axis]
            for unit, summary in expected.items():
                selected = assignments == unit
                summary['counts'] += np.histogram(times[selected], bins=edges)[0]
                for field in fields:
                    finite = selected & np.isfinite(scalars[field])
                    summary['trend_sums'][field] += np.histogram(
                        times[finite],
                        bins=edges,
                        weights=scalars[field][finite].astype(np.float64),
                    )[0]
                    summary['trend_counts'][field] += np.histogram(times[finite], bins=edges)[0]
        for unit, summary in expected.items():
            observed = actual[unit]
            np.testing.assert_array_equal(observed['counts'], summary['counts'])
            for field in fields:
                np.testing.assert_array_equal(
                    observed['trend_counts'][field], summary['trend_counts'][field]
                )
                np.testing.assert_allclose(
                    observed['trend_sums'][field],
                    summary['trend_sums'][field],
                    rtol=1e-9,
                    atol=1e-6,
                )
                means = np.full(bins, np.nan)
                np.divide(
                    summary['trend_sums'][field],
                    summary['trend_counts'][field],
                    out=means,
                    where=(summary['trend_counts'][field] > 0) & (observed['exposure'] > 0),
                )
                np.testing.assert_allclose(
                    observed['trends'][field], means, rtol=1e-9, atol=1e-6, equal_nan=True
                )

        return {
            'all_source_rows_checked': int(n_spikes),
            'batch_size': batch_size,
            'verified_unit_ids': list(actual),
            'counts_comparison': 'exact int64',
            'finite_counts_comparison': 'exact int64',
            'finite_sums_comparison': {'dtype': 'float64', 'rtol': 1e-9, 'atol': 1e-6},
            'seconds': time.perf_counter() - started,
        }
    finally:
        for array in arrays.values():
            close_array(array)


def capture(args):
    bounded = args.bounded.resolve()
    output = args.output.resolve()
    manifest = json.loads((bounded / '_phy_curation_manifest.json').read_text())
    source = Path(manifest['source']['path']).resolve()
    if source.name == 'params.py':
        source = source.parent
    if (
        bounded == output
        or bounded in output.parents
        or source == output
        or source in output.parents
    ):
        raise ValueError('--output must be outside both the bounded input and its source sorting.')
    output.mkdir(parents=True, exist_ok=False)
    before_package = _package_hashes(bounded)
    working = output / 'working-bounded'
    shutil.copytree(bounded, working, ignore=shutil.ignore_patterns('.phy', 'phy.log'))
    config = output / 'config'
    app = QApplication.instance() or QApplication(sys.argv[:1])
    controller = TemplateController(
        dir_path=working, config_dir=config, plugins=[], enable_threading=False
    )
    gui = None
    try:
        gui = controller.create_gui(
            default_views=['EvidenceAmplitudeView', 'UnitStabilityView'], do_prompt_save=False
        )
        service = controller._live_summaries
        service.suspend()
        session, supervisor = controller.bounded_session, controller.supervisor
        initial_events = json.loads(json.dumps(session.events))
        names = _required_arrays(manifest, initial_events)
        before_source = {name: hash_file(source / name) for name in names}
        required_bytes = sum((source / name).stat().st_size for name in names)
        retained = session.attributes['amplitudes.npy'].reshape(-1)[
            session.assignments == args.unit
        ]
        finite = retained[np.isfinite(retained)]
        if len(finite) < 2:
            raise ValueError('Selected unit requires at least two finite retained amplitudes.')
        threshold = float(np.median(finite))
        selected = np.isfinite(retained) & (retained < threshold)
        if not selected.any() or selected.all():
            raise ValueError('Median threshold must leave retained evidence in both children.')
        gui.resize(2200, 1250)
        gui.setWindowTitle(f'Live exact summaries — {PROVENANCE}')
        gui.show()
        service.suspend()
        amplitude, stability = (
            gui.get_view(name) for name in ('EvidenceAmplitudeView', 'UnitStabilityView')
        )
        for name in ('SimilarityView', 'MergePropositionsView', 'PropositionView'):
            for view in gui.list_views(name):
                view.dock.hide()
        gui.splitDockWidget(amplitude.dock, stability.dock, Qt.Horizontal)
        gui.resizeDocks([amplitude.dock, stability.dock], [1100, 1100], Qt.Horizontal)
        amplitude.amplitude_scope = 'overview'
        if args.rate_only:
            stability.toggle_amplitude(False)
            stability.toggle_depth(False)
        controls = [] if args.control is None else [args.control]
        supervisor.select([args.unit, *controls])
        supervisor.block()
        _capture(gui, output / 'before-split.png')
        split_started = time.perf_counter()
        update = supervisor.split_source_amplitude(args.unit, threshold)
        split_seconds = time.perf_counter() - split_started
        children = list(map(int, update.added))
        event = json.loads(json.dumps(session.events[-1]))
        supervisor.select([*children, *controls])
        supervisor.block()
        if any(controller.get_full_count(child) is not None for child in children):
            raise RuntimeError('Fresh split unexpectedly resolved without a live worker.')
        timing = {}

        def observe():
            now = time.perf_counter()
            if service._running is not None:
                timing.setdefault('job_started', now)
            if all(controller.get_full_count(child) is not None for child in children):
                timing.setdefault('summaries_ready', time.perf_counter())

        service.changed.connect(observe)
        resumed = time.perf_counter()
        service.resume()
        _wait(lambda: 'job_started' in timing, args.timeout, 'Worker did not start', service)
        recomputing_started = all(controller.get_full_count(child) is None for child in children)
        _capture(gui, output / 'recomputing.png')
        recomputing_finished = all(controller.get_full_count(child) is None for child in children)
        _wait(
            lambda: all(controller.get_full_count(child) is not None for child in children),
            args.timeout,
            'Full-source recomputation did not finish',
            service,
        )
        observe()
        supervisor.block()
        _capture(gui, output / 'after-split.png')
        events = json.loads(json.dumps(session.events))
        actual = {}
        native_edges = np.load(working / manifest['summaries']['edges_file'], allow_pickle=False)
        cached_lookup_started = time.perf_counter()
        for unit in [*children, *controls]:
            data = stability.get_data(unit, metrics=stability._metrics)
            np.testing.assert_array_equal(data.edges, native_edges / manifest['sample_rate'])
            actual[unit] = {
                'edges': native_edges,
                'counts': data.counts,
                'trend_sums': data.trend_sums,
                'trend_counts': data.trend_counts,
                'trends': data.trends,
                'exposure': data.exposure,
            }
            if int(data.counts.sum()) != controller.get_full_count(unit):
                raise AssertionError('GUI full count disagrees with its displayed summary.')
        cached_lookup_seconds = time.perf_counter() - cached_lookup_started
        oracle = _verify_source(source, manifest, events, actual, args.batch_size)
        after_source = {name: hash_file(source / name) for name in names}
        after_package = _package_hashes(bounded)
        if before_source != after_source or before_package != after_package:
            raise AssertionError('Original source arrays or bounded package changed.')
        report = {
            'format': 'phy_live_summary_capture',
            'format_version': 1,
            'capture_tool_sha256': hash_file(Path(__file__)),
            'capture_revision': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True
            ).strip(),
            'provenance': PROVENANCE,
            'dataset_label': args.dataset_label,
            'acceptance_scope': 'Existing converted engineering package; not new dataset acceptance',
            'input_package': str(bounded),
            'working_package': str(working),
            'source_directory': str(source),
            'source_binding': manifest['source'],
            'parent_unit': args.unit,
            'control_units': controls,
            'children': children,
            'threshold': threshold,
            'split_event': event,
            'events_sha256': hash_json(events),
            'initial_event_count': len(initial_events),
            'final_event_count': len(events),
            'native_bins': len(next(iter(actual.values()))['counts']),
            'source_spikes': manifest['num_spikes'],
            'retained_spikes': manifest['bounded_num_spikes'],
            'full_counts': {str(unit): int(data['counts'].sum()) for unit, data in actual.items()},
            'timing_seconds': {
                'bounded_split': split_seconds,
                'resume_to_worker_launch_notification': timing['job_started'] - resumed,
                'first_job_validation_scan_and_gui_delivery': timing['summaries_ready'] - resumed,
                'worker_notification_to_exact_results': timing['summaries_ready']
                - timing['job_started'],
                'resume_to_exact_results': timing['summaries_ready'] - resumed,
                'completed_cached_summary_lookup': cached_lookup_seconds,
            },
            'timing_context': 'Preflight SHA256 warms filesystem cache; not cold-storage latency',
            'source_required_bytes': required_bytes,
            'source_array_inventory': {
                name: manifest['source_array_inventory'][name] for name in names
            },
            'source_sha256_before': before_source,
            'source_sha256_after': after_source,
            'input_package_sha256_before': before_package,
            'input_package_sha256_after': after_package,
            'inputs_unchanged': True,
            'oracle': oracle,
            'rate_only': args.rate_only,
            'recomputing_capture_pending_at_entry': recomputing_started,
            'recomputing_capture_pending_at_exit': recomputing_finished,
            'recomputing_capture_note': 'A short job may finish during screenshot rendering; pending flags report observed state.',
            'screenshots': ['before-split.png', 'recomputing.png', 'after-split.png'],
            'gui_size': [gui.width(), gui.height()],
        }
        (output / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        return report
    finally:
        if gui is not None:
            _close_capture(gui, controller)
        else:
            controller.close()
        app.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bounded', required=True, type=Path)
    parser.add_argument(
        '--output', required=True, type=Path, help='Fresh exclusive output directory.'
    )
    parser.add_argument('--unit', type=int, default=49)
    parser.add_argument('--control', type=int, help='Optional unchanged comparison unit, e.g. 50.')
    parser.add_argument('--dataset-label', default='Existing converted engineering package')
    parser.add_argument('--rate-only', action='store_true')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--batch-size', type=int, default=100000)
    args = parser.parse_args()
    if args.timeout <= 0 or args.batch_size <= 0:
        parser.error('--timeout and --batch-size must be positive.')
    if args.unit == args.control:
        parser.error('--control must differ from --unit.')
    capture(args)


if __name__ == '__main__':
    main()
