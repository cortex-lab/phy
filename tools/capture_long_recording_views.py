"""Capture reproducible screenshots of the bounded-evidence Qt views.

The script builds a temporary synthetic Phy sorting, prepares it with the real
long-recording workflow, opens the real GUI, and saves screenshots. Existing
bounded inputs are copied into a temporary directory before opening.

Run with a display, or under a virtual X server::

    QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
      xvfb-run -a -s '-screen 0 2400x1600x24 +extension GLX' \
      uv run python tools/capture_long_recording_views.py --amplitude-split

Use ``--amplitude-split --bounded-dir PATH --cluster-ids ID`` for an existing
fresh bounded package, optionally with ``--threshold SOURCE_VALUE``. The three
PNG captures and ``capture.json`` include the exact committed threshold and
retained child sizes; graphical threshold values reflect pixel quantization.
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
from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, QPointF
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.source import hash_file
from phy.apps.template.gui import TemplateController
from phy.gui.qt import QApplication, Qt
from phy.plot.transform import NDC, Range

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / 'docs' / 'images'
SAMPLE_RATE = 30_000
HOURS = 8
CLUSTER_IDS = np.array([0, 2, 7, 99], dtype=np.int32)


def _make_source(path, hours=HOURS, unit_count=4):
    """Write a varied but deterministic synthetic source sorting."""
    path.mkdir(parents=True)
    rng = np.random.default_rng(20261013)
    cluster_ids = CLUSTER_IDS if unit_count == 4 else np.arange(unit_count, dtype=np.int32)
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

    unit_index = np.arange(len(spike_times), dtype=np.int64) % len(cluster_ids)
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
    np.save(path / 'spike_clusters.npy', cluster_ids[unit_index])
    np.save(path / 'spike_templates.npy', templates_for_spike)
    np.save(path / 'amplitudes.npy', amplitudes)
    np.save(path / 'spike_depths.npy', depths)

    channel_map = np.arange(4, dtype=np.int32)
    channel_positions = np.array([[0, 0], [20, 100], [0, 200], [20, 300]], dtype=np.float32)
    np.save(path / 'channel_map.npy', channel_map)
    np.save(path / 'channel_positions.npy', channel_positions)

    templates = np.zeros((len(cluster_ids), 40, len(channel_map)), dtype=np.float32)
    time = np.arange(40, dtype=np.float32)
    for unit in range(len(cluster_ids)):
        center = 13 + (unit % len(channel_map)) * 3
        pulse = -np.exp(-0.5 * ((time - center) / 3.5) ** 2) * (35 + 8 * unit)
        for channel in range(len(channel_map)):
            templates[unit, :, channel] = pulse * np.exp(
                -0.7 * abs(channel - unit % len(channel_map))
            )
    np.save(path / 'templates.npy', templates)

    pc_feature_ind = np.array(
        [
            [unit % len(channel_map), (unit + 1) % len(channel_map), (unit + 2) % len(channel_map)]
            for unit in range(len(cluster_ids))
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
        stream.write('cluster_id\tgroup\n')
        for unit in cluster_ids:
            stream.write(f'{unit}\t{"mua" if unit == 7 else "good"}\n')
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
        assert all(ax.get_legend() is None for ax in view.canvas.iter_ax())
        _capture(gui, output_dir / 'after-overview.png')
        started = time.perf_counter()
        view.toggle_amplitude(False)
        view.toggle_depth(False)
        rate_only_seconds = time.perf_counter() - started
        _capture(gui, output_dir / 'after-rate-only.png')
        camera = view.screenshot(dir=output_dir / 'camera')
        shutil.copyfile(camera, output_dir / 'camera-rate-only.png')
        native = float(np.diff(view.get_data(selected[0]).edges)[0])
        view.set_display_bin_seconds(2 * native)
        _capture(gui, output_dir / 'after-rate-coarsened.png')
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
            'displayed_cluster_ids': list(map(int, view.displayed_ids)),
            'merge_cluster_ids': supervisor.selected_merge,
            'summary_bins': len(view.get_data(selected[0]).exposure),
            'full_counts': {
                str(unit): int(view.get_data(unit).counts.sum()) for unit in view.displayed_ids
            },
            'timing_seconds': {
                'select_and_block': selection_seconds,
                'rate_only_redraws': rate_only_seconds,
            },
            'gui_size': [gui.width(), gui.height()],
            'display_bins': len(view.get_display_data(selected[0]).exposure),
            'rate_line_width': view.rate_line_width,
            'display_limit': view.display_limit,
            'display_bin_seconds': view.display_bin_seconds,
            'screenshots': [
                'after-overview.png',
                'after-rate-only.png',
                'after-merge.png',
                'after-rate-coarsened.png',
                'camera-rate-only.png',
            ],
            'source_arrays_modified': False,
        }
        (output_dir / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        _close_capture(gui, controller)


def _capture_amplitude_split(bounded, config_dir, output_dir, selected, threshold, dataset_label):
    """Capture a real threshold drag and K commit on a disposable bounded copy."""
    controller = TemplateController(
        dir_path=bounded, config_dir=config_dir, plugins=[], enable_threading=False
    )
    gui = controller.create_gui(
        default_views=['EvidenceAmplitudeView', 'WaveformView'], do_prompt_save=False
    )
    try:
        supervisor = controller.supervisor
        if controller.bounded_session.events:
            raise ValueError('Amplitude capture requires a fresh bounded package without edits.')
        if len(selected) != 1:
            raise ValueError('Amplitude capture requires exactly one cluster ID.')
        unit = selected[0]
        ids = supervisor.clustering.spikes_per_cluster.get(unit, np.array([], dtype=np.int64))
        values = controller.bounded_session.attributes.get('amplitudes.npy')
        if values is None or not len(ids):
            raise ValueError('Select a retained unit with source amplitude evidence.')
        amplitudes = values.reshape(-1)[ids]
        finite = amplitudes[np.isfinite(amplitudes)]
        if not len(finite):
            raise ValueError('The selected unit has no finite source amplitudes.')
        threshold = float(np.median(finite)) if threshold is None else float(threshold)
        if not np.isfinite(threshold):
            raise ValueError('The threshold must be finite.')
        if not np.any(np.isfinite(amplitudes) & (amplitudes < threshold)) or not np.any(
            ~(np.isfinite(amplitudes) & (amplitudes < threshold))
        ):
            raise ValueError('Both threshold children must have retained evidence.')

        gui.resize(2200, 1250)
        gui.show()
        _wait_for_render()
        for name in ('ClusterView', 'SimilarityView', 'MergePropositionsView', 'PropositionView'):
            for view in gui.list_views(name):
                view.dock.hide()
        amplitude = gui.get_view('EvidenceAmplitudeView')
        waveform = gui.get_view('WaveformView')
        gui.splitDockWidget(amplitude.dock, waveform.dock, Qt.Horizontal)
        gui.resizeDocks([amplitude.dock, waveform.dock], [1450, 700], Qt.Horizontal)
        controller.n_spikes_waveforms = 60
        controller.n_spikes_waveforms_total = 60
        supervisor.select(selected)
        supervisor.block()
        amplitude.amplitude_scope = 'overview'
        waveform.waveforms_type = 'waveforms'
        waveform.plot()
        _wait_for_render()
        _capture(gui, output_dir / 'amplitude-before.png')

        # Deliver actual Qt events to the native OpenGL window. Reuse the view's
        # transform so the drag is expressed in unscaled source amplitude units.
        canvas = amplitude.canvas
        width, height = canvas.get_size()
        ndc = Range(amplitude.data_bounds, NDC).apply([[amplitude.duration / 2, threshold]])[0]
        screen = (ndc + np.asarray(canvas.panzoom.pan)) * canvas.panzoom._zoom_aspect()
        point = QPoint(round((screen[0] + 1) * width / 2), round((1 - screen[1]) * height / 2))
        if not 0 <= point.y() < height:
            raise ValueError('The threshold is outside the displayed amplitude range.')
        start = QPoint(max(0, point.x() - 20), point.y())
        QTest.mousePress(canvas, Qt.RightButton, Qt.AltModifier, start)
        move = QMouseEvent(
            QEvent.MouseMove, QPointF(point), Qt.NoButton, Qt.RightButton, Qt.AltModifier
        )
        QApplication.sendEvent(canvas, move)
        QTest.mouseRelease(canvas, Qt.RightButton, Qt.AltModifier, point)
        _wait_for_render()
        actual_threshold = amplitude.split_threshold
        if actual_threshold is None or not supervisor.actions.get('split').isEnabled():
            raise RuntimeError('The actual Alt-right drag did not create an eligible preview.')
        _capture(gui, output_dir / 'amplitude-preview.png')
        highlighted = int(len(waveform._highlighted_spike_ids))
        if hasattr(canvas, 'requestActivate'):
            canvas.requestActivate()
        elif hasattr(canvas, 'setFocus'):
            canvas.setFocus()
        _wait_for_render()
        QTest.keyClick(canvas, Qt.Key_K)
        supervisor.block()
        _wait_for_render()
        if not controller.bounded_session.events:
            raise RuntimeError('The actual K key did not commit the threshold predicate.')
        event = controller.bounded_session.events[-1]
        if event['operation'] != 'threshold_split' or amplitude.split_threshold is not None:
            raise RuntimeError('K did not finish the expected semantic threshold split.')
        _capture(gui, output_dir / 'amplitude-after.png')
        report = {
            'dataset_label': dataset_label,
            'workflow': 'amplitude_split',
            'selected_cluster_ids': selected,
            'amplitude_scope': amplitude.amplitude_scope,
            'requested_source_threshold': threshold,
            'exact_source_threshold': event['predicate']['threshold'],
            'predicate': event['predicate'],
            'retained_parent_count': int(len(ids)),
            'retained_split_sizes': {
                str(child): int(supervisor.n_spikes(child)) for child in event['children']
            },
            'highlighted_waveform_traces': highlighted,
            'gui_size': [gui.width(), gui.height()],
            'screenshots': [
                'amplitude-before.png',
                'amplitude-preview.png',
                'amplitude-after.png',
            ],
            'source_arrays_modified': False,
            'capture_revision': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True
            ).strip(),
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        '--comparison', action='store_true', help='Capture cluster overlay controls.'
    )
    modes.add_argument(
        '--amplitude-split',
        action='store_true',
        help='Capture a source-amplitude drag and K split.',
    )
    parser.add_argument(
        '--threshold', type=float, help='Source amplitude threshold; default: median.'
    )
    parser.add_argument(
        '--cluster-ids',
        type=int,
        nargs='+',
        help='Selected IDs; exactly one for amplitude splits.',
    )
    parser.add_argument('--dataset-label', help='Dataset provenance label stored in capture.json.')
    parser.add_argument('--synthetic-hours', type=int, default=HOURS)
    parser.add_argument('--synthetic-units', type=int, default=4)
    parser.add_argument('--summary-bin-seconds', type=float, default=1800)
    args = parser.parse_args()
    if args.synthetic_hours < 4:
        parser.error('--synthetic-hours must be at least 4.')
    if args.synthetic_units < 2:
        parser.error('--synthetic-units must be at least 2.')
    if args.bounded_dir and not (args.comparison or args.amplitude_split):
        parser.error('--bounded-dir requires --comparison or --amplitude-split.')
    if args.threshold is not None and not args.amplitude_split:
        parser.error('--threshold requires --amplitude-split.')
    if args.amplitude_split and args.cluster_ids and len(args.cluster_ids) != 1:
        parser.error('--amplitude-split requires exactly one --cluster-ids ID.')
    if args.bounded_dir:
        source = args.bounded_dir.resolve()
        if not (source / '_phy_curation_manifest.json').is_file():
            parser.error('--bounded-dir must contain a bounded curation manifest.')
        if source == args.output_dir.resolve() or source in args.output_dir.resolve().parents:
            parser.error('--output-dir must be outside the input bounded package.')
        if args.amplitude_split:
            if not args.cluster_ids or len(args.cluster_ids) != 1:
                parser.error(
                    '--bounded-dir amplitude splits require exactly one --cluster-ids ID.'
                )
        elif not args.cluster_ids or len(args.cluster_ids) < 2:
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
            capture = _capture_amplitude_split if args.amplitude_split else _capture_comparison
            extra = [args.threshold] if args.amplitude_split else []
            capture(
                bounded,
                root / 'config',
                args.output_dir,
                args.cluster_ids,
                *extra,
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
        params, intervals = _make_source(
            root / 'source', hours=args.synthetic_hours, unit_count=args.synthetic_units
        )
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
            bin_seconds=args.summary_bin_seconds,
        )
        if args.amplitude_split:
            _capture_amplitude_split(
                bounded,
                root / 'config',
                args.output_dir,
                args.cluster_ids or [0],
                args.threshold,
                args.dataset_label or f'Synthetic {args.synthetic_hours}-hour fixture.',
            )
        elif args.comparison:
            _capture_comparison(
                bounded,
                root / 'config',
                args.output_dir,
                args.cluster_ids
                or (
                    list(map(int, CLUSTER_IDS[:3]))
                    if args.synthetic_units == 4
                    else list(range(min(12, args.synthetic_units)))
                ),
                args.dataset_label or f'Synthetic {args.synthetic_hours}-hour fixture.',
            )
        else:
            _capture_views(bounded, root / 'config-stability', args.output_dir, stability=True)
            _capture_views(bounded, root / 'config-evidence', args.output_dir, stability=False)
    app.quit()


if __name__ == '__main__':
    main()
