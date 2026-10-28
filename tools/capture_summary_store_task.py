"""Capture actual GUI store preparation and measure the Qt heartbeat on a copy.

Original inputs are read-only; the output includes a copied bounded package and
new external store. This is a GUI responsiveness trial, not cold-storage timing.
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

from phy.apps.template.gui import TemplateController
from phy.gui.qt import QApplication, QTimer


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    package = args.output / 'bounded'
    shutil.copytree(
        args.bounded,
        package,
        ignore=shutil.ignore_patterns('live-summary-cache', 'live-summary-store.json'),
    )
    app = QApplication.instance() or QApplication(sys.argv[:1])
    controller = TemplateController(
        dir_path=package, config_dir=args.output / 'config', plugins=[]
    )
    gui = controller.create_gui(default_views=['UnitStabilityView'], do_prompt_save=False)
    actions = controller._summary_store_actions
    actions._checked = True
    gui.resize(2200, 1250)
    gui.setWindowTitle('Summary-store preparation — converted engineering dataset')
    gui.show()
    controller.supervisor.select([49, 50])
    controller.supervisor.block()
    actions.build_dialog()
    captures, samples, phases = [], [], []

    def capture(name):
        started = time.perf_counter()
        _capture(gui, args.output / name)
        captures.append((started, time.perf_counter()))

    def wait_until(predicate, timeout=180):
        deadline = time.perf_counter() + timeout
        while not predicate():
            if time.perf_counter() > deadline:
                raise TimeoutError('GUI summary task did not complete in time.')
            app.processEvents()
            time.sleep(0.001)

    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: samples.append(time.perf_counter()))
    actions.manager.progress.connect(lambda p: phases.append((time.perf_counter(), p.stage)))
    try:
        wait_until(lambda: actions.dialog is not None)
        actions.dialog.destination.setText(str(args.output / 'store'))
        wait_until(actions.dialog.build_button.isEnabled)
        capture('prompt.png')
        timer.start()
        started = time.perf_counter()
        actions.dialog._build()
        children = list(
            map(int, controller.supervisor.split_source_amplitude(49, -341.797119140625).added)
        )
        controller.supervisor.select([*children, 50])
        controller.supervisor.block()
        wait_until(lambda: any(stage == 'Building summary store' for _, stage in phases))
        # Allow measured throughput to stabilize before capturing the ETA.
        eta_ready = time.perf_counter() + 3
        wait_until(lambda: time.perf_counter() >= eta_ready)
        capture('building.png')
        wait_until(lambda: not actions.manager.busy and actions._mode == 'attach')
        wait_until(lambda: all(controller.get_full_count(u) is not None for u in children))
        duration = time.perf_counter() - started
        timer.stop()
        capture('ready.png')
        intervals = [
            (a, b, b - a)
            for a, b in zip(samples[:-1], samples[1:])
            if not any(a <= end and b >= first for first, end in captures)
        ]
        values = np.array([item[2] for item in intervals])
        report = {
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_spikes': controller.bounded_session.manifest['num_spikes'],
            'label': 'Earlier converted 112.859h engineering dataset; not native1000h acceptance',
            'build_attach_and_child_summary_seconds': duration,
            'heartbeat': {
                'target_interval_ms': 10,
                'samples': len(values),
                'p50_ms': float(np.quantile(values, 0.5) * 1000),
                'p95_ms': float(np.quantile(values, 0.95) * 1000),
                'max_ms': float(values.max() * 1000),
                'screenshot_intervals_excluded': True,
            },
            'children': children,
            'child_counts': [int(controller.get_full_count(u)) for u in children],
            'curation_remained_unsaved': controller.supervisor.is_dirty(),
            'stages': list(dict.fromkeys(stage for _, stage in phases)),
            'screenshots': ['prompt.png', 'building.png', 'ready.png'],
        }
        (args.output / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        timer.stop()
        if actions.manager.busy:
            actions.manager.cancel()
            wait_until(lambda: not actions.manager.busy)
        _close_capture(gui, controller)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bounded', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
