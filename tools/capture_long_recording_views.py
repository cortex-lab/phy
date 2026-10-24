"""Capture reproducible screenshots of the bounded-evidence Qt views.

The script builds a temporary synthetic Phy sorting, prepares it with the real
long-recording workflow, opens the real GUI, and saves two screenshots. It does
not download or modify a recording.

Run with a display, or under a virtual X server::

    QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
      xvfb-run -a -s '-screen 0 2400x1600x24 +extension GLX' \
      uv run python tools/capture_long_recording_views.py
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtTest import QTest

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.source import hash_file
from phy.apps.template.gui import TemplateController
from phy.gui.qt import QApplication, Qt

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / 'docs' / 'images'
SAMPLE_RATE = 30_000
HOURS = 8
CLUSTER_IDS = np.array([0, 2, 7, 99], dtype=np.int32)


def _make_source(path, hours=HOURS):
    """Write a varied but deterministic synthetic source sorting."""
    path.mkdir(parents=True)
    rng = np.random.default_rng(20261013)
    samples = SAMPLE_RATE * 3600 * hours
    intervals = np.array(
        [[0, 3 * 3600 * SAMPLE_RATE], [int(3.5 * 3600 * SAMPLE_RATE), samples]],
        dtype=np.int64,
    )
    valid = intervals[:, 1] - intervals[:, 0]
    cumulative = np.cumsum(valid)
    offsets = rng.integers(0, cumulative[-1], size=120_000, dtype=np.int64)
    interval_index = np.searchsorted(cumulative, offsets, side='right')
    previous = np.where(interval_index == 0, 0, cumulative[np.maximum(interval_index - 1, 0)])
    spike_times = intervals[interval_index, 0] + offsets - previous
    order = np.argsort(spike_times, kind='stable')
    spike_times = spike_times[order]

    unit_index = np.arange(len(spike_times), dtype=np.int64) % len(CLUSTER_IDS)
    unit_index = unit_index[order]
    elapsed = spike_times / SAMPLE_RATE
    hour_phase = elapsed / 3600
    amplitudes = (
        35
        + unit_index * 9
        + 5 * np.sin(hour_phase * np.pi / 3)
        + 2 * hour_phase
        + rng.normal(0, 2.5, len(spike_times))
    ).astype(np.float32)
    depths = (
        250
        + unit_index * 180
        + 20 * np.sin(hour_phase * np.pi / 2)
        + rng.normal(0, 8, len(spike_times))
    ).astype(np.float32)
    templates_for_spike = unit_index.astype(np.int32)

    np.save(path / 'spike_times.npy', spike_times.astype(np.int64))
    np.save(path / 'spike_clusters.npy', CLUSTER_IDS[unit_index])
    np.save(path / 'spike_templates.npy', templates_for_spike)
    np.save(path / 'amplitudes.npy', amplitudes)
    np.save(path / 'spike_depths.npy', depths)

    channel_map = np.arange(4, dtype=np.int32)
    channel_positions = np.array([[0, 0], [20, 100], [0, 200], [20, 300]], dtype=np.float32)
    np.save(path / 'channel_map.npy', channel_map)
    np.save(path / 'channel_positions.npy', channel_positions)

    templates = np.zeros((len(CLUSTER_IDS), 40, len(channel_map)), dtype=np.float32)
    time = np.arange(40, dtype=np.float32)
    for unit in range(len(CLUSTER_IDS)):
        center = 13 + unit * 3
        pulse = -np.exp(-0.5 * ((time - center) / 3.5) ** 2) * (35 + 8 * unit)
        for channel in range(len(channel_map)):
            templates[unit, :, channel] = pulse * np.exp(-0.7 * abs(channel - unit))
    np.save(path / 'templates.npy', templates)

    pc_feature_ind = np.array(
        [
            [unit, (unit + 1) % len(channel_map), (unit + 2) % len(channel_map)]
            for unit in range(len(CLUSTER_IDS))
        ],
        dtype=np.int32,
    )
    np.save(path / 'pc_feature_ind.npy', pc_feature_ind)
    features = np.empty((len(spike_times), 3, 3), dtype=np.float32)
    features[:, 0, :] = unit_index[:, None] * 3.5 + rng.normal(0, 0.65, (len(spike_times), 3))
    features[:, 1, :] = np.sin(hour_phase[:, None] * np.pi / 2) + rng.normal(
        0, 0.7, (len(spike_times), 3)
    )
    features[:, 2, :] = np.cos(hour_phase[:, None] * np.pi / 3) + rng.normal(
        0, 0.6, (len(spike_times), 3)
    )
    np.save(path / 'pc_features.npy', features)

    # Stored waveforms are a display reservoir, aligned to sorted source rows.
    waveform_ids = np.arange(0, len(spike_times), 8, dtype=np.int64)
    waveform_channels = np.tile(
        np.arange(len(channel_map), dtype=np.int32), (len(waveform_ids), 1)
    )
    waveforms = templates[unit_index[waveform_ids]].copy()
    waveforms += rng.normal(0, 2.2, waveforms.shape).astype(np.float32)
    stem = '_phy_spikes_subset'
    np.save(path / f'{stem}.spikes.npy', waveform_ids)
    np.save(path / f'{stem}.channels.npy', waveform_channels)
    np.save(path / f'{stem}.waveforms.npy', waveforms)

    with (path / 'cluster_group.tsv').open('w', encoding='utf-8') as stream:
        stream.write('cluster_id\tgroup\n0\tgood\n2\tgood\n7\tmua\n99\tgood\n')
    (path / 'params.py').write_text(
        f'sample_rate = {SAMPLE_RATE}.0\n'
        f'num_samples = {samples}\n'
        'n_channels_dat = 4\n'
        "dtype = 'int16'\n"
        'dat_path = None\n'
        f'valid_intervals = {intervals.tolist()!r}\n',
        encoding='utf-8',
    )
    return path / 'params.py', intervals


def _wait_for_render():
    QApplication.processEvents()
    QTest.qWait(1200)
    QApplication.processEvents()


def _capture(gui, destination):
    _wait_for_render()
    # Phy's OpenGL canvases are native child windows. QWidget.grab() omits
    # those children on some Qt/X11 combinations, so grab the rendered screen
    # and crop to the application window instead.
    screen = QApplication.primaryScreen()
    frame = gui.frameGeometry()
    pixmap = screen.grabWindow(0).copy(frame.x(), frame.y(), frame.width(), frame.height())
    if not pixmap.save(str(destination), 'PNG'):
        raise OSError(f'Could not save screenshot: {destination}')


def _close_capture(gui, controller):
    gui.close()
    controller.close()
    gui.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()


def _capture_comparison(bounded, config_dir, output_dir, selected, dataset_label):
    """Capture actual GUI comparisons on an isolated, disposable package."""
    controller = TemplateController(
        dir_path=bounded, config_dir=config_dir, plugins=[], enable_threading=False
    )
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    try:
        gui.resize(2100, 1200)
        gui.show()
        _wait_for_render()
        supervisor = controller.supervisor
        started = time.perf_counter()
        supervisor.select(selected)
        supervisor.block()
        selection_seconds = time.perf_counter() - started
        view = gui.get_view('UnitStabilityView')
        assert list(view.cluster_ids) == supervisor.selected
        assert len(view.cluster_ids) >= 2, 'Select at least two retained clusters.'
        _capture(gui, output_dir / 'after-overview.png')
        started = time.perf_counter()
        view.toggle_amplitude(False)
        view.toggle_depth(False)
        rate_only_seconds = time.perf_counter() - started
        _capture(gui, output_dir / 'after-rate-only.png')
        supervisor.toggle_merge_mode()
        supervisor.block()
        assert supervisor.selection.state.is_merge_mode
        assert list(view.cluster_ids) == supervisor.selected
        _capture(gui, output_dir / 'after-merge.png')
        manifest = controller.bounded_session.manifest
        report = {
            'dataset_label': dataset_label,
            'capture_revision': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True
            ).strip(),
            'clock_hours': manifest['num_samples'] / manifest['sample_rate'] / 3600,
            'selected_cluster_ids': selected,
            'displayed_cluster_ids': list(map(int, view.cluster_ids[: view.display_limit])),
            'merge_cluster_ids': supervisor.selected_merge,
            'summary_bins': len(view.get_data(selected[0]).exposure),
            'full_counts': {
                str(unit): int(view.get_data(unit).counts.sum())
                for unit in view.cluster_ids[: view.display_limit]
            },
            'timing_seconds': {
                'select_and_block': selection_seconds,
                'rate_only_redraws': rate_only_seconds,
            },
            'gui_size': [gui.width(), gui.height()],
            'screenshots': ['after-overview.png', 'after-rate-only.png', 'after-merge.png'],
            'source_arrays_modified': False,
        }
        (output_dir / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        _close_capture(gui, controller)


def _capture_views(bounded, config_dir, output_dir, stability):
    views = (
        ['UnitStabilityView']
        if stability
        else [
            'EvidenceAmplitudeView',
            'FeatureView',
            'SampledISIView',
            'SampledCorrelogramView',
        ]
    )
    controller = TemplateController(
        dir_path=bounded,
        config_dir=config_dir,
        plugins=[],
        enable_threading=False,
    )
    gui = controller.create_gui(default_views=views, do_prompt_save=False)
    try:
        gui.resize(2100, 1200)
        gui.show()
        _wait_for_render()
        # Focus the captures on the plotted evidence rather than the GUI tables.
        similarity = gui.get_view('SimilarityView')
        if similarity is not None:
            similarity.dock.hide()
        cluster_table = gui.get_view('ClusterView')
        if cluster_table is not None:
            cluster_table.dock.hide()
        if not stability:
            amplitude, feature, isi, correlogram = (
                gui.get_view(name)
                for name in (
                    'EvidenceAmplitudeView',
                    'FeatureView',
                    'SampledISIView',
                    'SampledCorrelogramView',
                )
            )
            gui.splitDockWidget(amplitude.dock, feature.dock, Qt.Vertical)
            gui.splitDockWidget(amplitude.dock, isi.dock, Qt.Horizontal)
            gui.splitDockWidget(isi.dock, correlogram.dock, Qt.Vertical)
            gui.resizeDocks([amplitude.dock, isi.dock], [1000, 1000], Qt.Horizontal)
            gui.resizeDocks([amplitude.dock, feature.dock], [550, 550], Qt.Vertical)
            gui.resizeDocks([isi.dock, correlogram.dock], [550, 550], Qt.Vertical)
        _wait_for_render()
        windows = controller.evidence_windows
        times = controller.model.spike_samples
        counts = [
            int(np.searchsorted(times, end) - np.searchsorted(times, start))
            for start, end in windows
        ]
        best_window = int(np.argmax(counts))
        controller.inspect_evidence_window(best_window)
        selected = [0] if stability else [0, 2]
        controller.supervisor.select(selected)
        controller.supervisor.block()
        _wait_for_render()
        if stability:
            filename = 'long-recording-overview.png'
        else:
            filename = 'long-recording-evidence.png'
        destination = output_dir / filename
        _capture(gui, destination)
        print(
            f'{destination} ({gui.width()}x{gui.height()}, window {best_window}, '
            f'{counts[best_window]} evidence spikes)'
        )
    finally:
        _close_capture(gui, controller)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--bounded-dir', type=Path, help='Existing package; copied before opening.'
    )
    parser.add_argument(
        '--comparison', action='store_true', help='Capture cluster overlay controls.'
    )
    parser.add_argument('--cluster-ids', type=int, nargs='+', help='IDs for comparison captures.')
    parser.add_argument('--dataset-label', help='Dataset provenance label stored in capture.json.')
    parser.add_argument('--synthetic-hours', type=int, default=HOURS)
    args = parser.parse_args()
    if args.synthetic_hours < 4:
        parser.error('--synthetic-hours must be at least 4.')
    if args.bounded_dir and not args.comparison:
        parser.error('--bounded-dir requires --comparison.')
    if args.bounded_dir:
        source = args.bounded_dir.resolve()
        if not (source / '_phy_curation_manifest.json').is_file():
            parser.error('--bounded-dir must contain a bounded curation manifest.')
        if source == args.output_dir.resolve() or source in args.output_dir.resolve().parents:
            parser.error('--output-dir must be outside the input bounded package.')
        if not args.cluster_ids or len(args.cluster_ids) < 2:
            parser.error('--bounded-dir comparisons require at least two --cluster-ids.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv[:1])

    with tempfile.TemporaryDirectory(prefix='phy-long-recording-screenshots-') as temporary:
        root = Path(temporary)
        if args.bounded_dir:
            bounded = root / 'bounded'
            original_hashes = {
                str(path.relative_to(source)): hash_file(path)
                for path in source.rglob('*')
                if path.is_file()
            }
            shutil.copytree(source, bounded, ignore=shutil.ignore_patterns('.phy', '*.log'))
            _capture_comparison(
                bounded,
                root / 'config',
                args.output_dir,
                args.cluster_ids,
                args.dataset_label
                or 'Existing bounded package; acquisition exposure as declared.',
            )
            checked_hashes = {
                str(path.relative_to(source)): hash_file(path)
                for path in source.rglob('*')
                if path.is_file()
            }
            if original_hashes != checked_hashes:
                raise RuntimeError('The input bounded package changed during capture.')
            report_path = args.output_dir / 'capture.json'
            report = json.loads(report_path.read_text())
            report['input_sha256'] = original_hashes
            report['input_package_unchanged'] = True
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            app.quit()
            return
        params, intervals = _make_source(root / 'source', hours=args.synthetic_hours)
        bounded = prepare_large(
            params,
            root / 'bounded',
            duration_seconds=args.synthetic_hours * 3600,
            valid_intervals=intervals,
            max_spikes=20_000,
            max_bytes=268_435_456,
            window_seconds=30,
            stratum_seconds=3600,
            seed=0,
            batch_size=50_000,
            bin_seconds=1800,
        )
        if args.comparison:
            _capture_comparison(
                bounded,
                root / 'config',
                args.output_dir,
                args.cluster_ids or list(map(int, CLUSTER_IDS[:3])),
                args.dataset_label or f'Synthetic {args.synthetic_hours}-hour fixture.',
            )
        else:
            _capture_views(bounded, root / 'config-stability', args.output_dir, stability=True)
            _capture_views(bounded, root / 'config-evidence', args.output_dir, stability=False)
    app.quit()


if __name__ == '__main__':
    main()
