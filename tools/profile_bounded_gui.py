"""Profile real bounded GUI actions on a guarded copy under a private Xvfb.

Example: uv run --no-sync python tools/profile_bounded_gui.py --bounded PACKAGE
--output NEW_DIRECTORY --events 213 --merges 100 --profile

Requires Xvfb and glxinfo. Never opens the input in a controller. The copy budget
includes caches and old generations; raise it explicitly for larger packages.
GUI readiness means the supervisor queue and view busy flags have drained plus
an additional Qt timer turn; it is not a presentation/compositor timing claim.
"""

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import os
import platform
import select
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _inventory(source, maximum):
    """Reject aliases and unexpected file types before copying or hashing."""
    files = []
    size = 0
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symbolic links are not supported: {path}')
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f'Not a regular file: {path}')
        size += path.stat().st_size
        if size > maximum:
            raise ValueError(f'Package exceeds --max-copy-bytes ({maximum:,}).')
        files.append(path.relative_to(source))
    return files, size


def _hashes(source, files):
    hashes = {}
    for relative in files:
        digest = hashlib.sha256()
        with (source / relative).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        hashes[str(relative)] = digest.hexdigest()
    return hashes


def _validate(args):
    source, output = args.bounded.resolve(), args.output.resolve()
    if not source.is_dir() or not (source / '_phy_curation_manifest.json').is_file():
        raise ValueError('--bounded must be an existing prepared bounded package.')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Input and output must be separate, non-nested directories.')
    if output.exists():
        raise FileExistsError('The output must be a new directory.')
    if args.repeats < 1 or args.merges < 0 or args.max_copy_bytes < 1 or args.timeout < 1:
        raise ValueError('Require positive repeats, timeout, copy budget and nonnegative merges.')
    if args.events is not None and args.events < args.merges:
        raise ValueError('--events must include the requested merge prefix.')
    if args.merges and args.events is None:
        raise ValueError('--merges requires --events.')
    return source, output


def _seed(package, events, merges):
    from phy.apps.large_recording.curation import BoundedSession, interpret_events

    with BoundedSession(package) as session:
        initial = len(session.events)
        if events is not None:
            live, _, _ = interpret_events(session.unit_ids, session.base_metadata, session.events)
            ids = sorted(live)
            if merges >= len(ids) - 1:
                raise ValueError('Seed merges must leave a separate unit for measured actions.')
            journal = list(session.events)
            target = ids[0]
            allocator = session.next_cluster_id
            for index in range(merges):
                journal.append(
                    {
                        'version': 2,
                        'operation': 'merge',
                        'parents': [target, ids[index + 1]],
                        'target': allocator,
                        'metadata': dict.fromkeys(set(session.metadata) | {'group'}),
                    }
                )
                target, allocator = allocator, allocator + 1
            for index in range(events - merges):
                journal.append(
                    {
                        'version': 2,
                        'operation': 'label',
                        'field': 'group',
                        'ids': [ids[-1]],
                        'value': 'mua' if index % 2 else 'good',
                    }
                )
            session.restore_journal({'events': tuple(journal), 'next_cluster_id': allocator})
            session.save()
        return {
            'original_events': initial,
            'initial_events': len(session.events),
            'synthetic_events_added': events or 0,
            'synthetic_merges_added': merges,
        }


def _worker(args):
    # Import order matches the application/capture harness: OpenGL before Qt.
    import OpenGL.GL as GL
    from PyQt5.QtCore import QTimer
    from PyQt5.QtTest import QTest

    from phy.apps.template.gui import TemplateController
    from phy.gui.qt import QApplication

    output = args.output
    report = {'measurements': [], 'status': 'failed'}
    controller = gui = None
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        report.update(_seed(output / 'working-package', args.events, args.merges))
        app = QApplication.instance() or QApplication([])
        setup = time.perf_counter()
        controller = TemplateController(
            dir_path=output / 'working-package',
            config_dir=output / 'config',
            plugins=[],
            enable_threading=not args.no_threading,
        )
        names = ['EvidenceAmplitudeView', 'UnitStabilityView']
        gui = controller.create_gui(
            default_views=names, view_count=dict.fromkeys(names, 1), do_prompt_save=False
        )
        gui.resize(1280, 900)
        gui.show()
        supervisor = controller.supervisor
        supervisor.set_skip_masked_clusters(False)

        def ready(block=True):
            if block:
                supervisor.block()
            app.processEvents()
            tick = []
            QTimer.singleShot(0, lambda: tick.append(True))
            while not tick:
                app.processEvents()
            if supervisor._is_busy or not supervisor.task_logger.has_finished():
                raise RuntimeError('GUI did not reach the documented ready condition.')

        ready()
        if any(gui.get_view(name) is None for name in names):
            raise RuntimeError('A requested default scientific view is missing.')
        report['gui_setup_seconds'] = time.perf_counter() - setup
        report['views'] = names
        report['threading'] = controller._enable_threading
        report['requested_threading'] = not args.no_threading
        report['ready_definition'] = 'supervisor.block + processEvents + zero-delay Qt timer'
        report['versions'] = {
            name: importlib.metadata.version(name)
            for name in ('phy', 'phylib', 'numpy', 'PyQt5', 'PyOpenGL')
        }
        # Query an actual shown scientific canvas, not the unbound process context.
        canvas = gui.get_view(names[0]).canvas
        canvas.makeCurrent()
        report['opengl'] = {
            name: str(GL.glGetString(value))
            for name, value in [
                ('vendor', GL.GL_VENDOR),
                ('renderer', GL.GL_RENDERER),
                ('version', GL.GL_VERSION),
            ]
        }
        report['python'] = sys.version
        report['platform'] = platform.platform()
        heartbeat = QTimer()
        heartbeat.setInterval(10)
        beats = []
        heartbeat.timeout.connect(lambda: beats.append(time.perf_counter()))
        heartbeat.start()
        QTest.qWait(30)

        def measure(name, operation, profiled=False):
            before = list(supervisor.selected_clusters)
            initial_events = len(controller.bounded_session.events)
            beats.clear()
            start, start_cpu = time.perf_counter(), time.process_time()
            beats.append(start)
            profiler = cProfile.Profile() if profiled else None
            if profiler:
                profiler.enable()
            operation()
            returned = time.perf_counter()
            supervisor.block()
            blocked = time.perf_counter()
            ready(block=False)
            ended = time.perf_counter()
            if profiler:
                profiler.disable()
                profiler.dump_stats(output / f'{name}.prof')
            gaps = [b - a for a, b in zip(beats, beats[1:] + [ended])]
            row = {
                'action': name,
                'profiled': profiled,
                'initial_events': initial_events,
                'selected_before': before,
                'selected_after': list(supervisor.selected_clusters),
                'action_return_seconds': returned - start,
                'supervisor_wait_seconds': blocked - returned,
                'additional_ready_seconds': ended - blocked,
                'wall_seconds': ended - start,
                'cpu_seconds': time.process_time() - start_cpu,
                'heartbeat_interval_seconds': 0.01,
                'heartbeat_ticks': len(beats) - 1,
                'heartbeat_max_gap_seconds': max(gaps, default=0),
                'heartbeat_max_delay_seconds': max(0, max(gaps, default=0) - 0.01),
            }
            report['measurements'].append(row)
            (output / 'worker-result.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            return row

        actions = supervisor.action_creator.edit_actions
        for repeat in range(args.repeats):
            ids = supervisor.cluster_view.get_navigable_ids()
            if len(ids) < 2:
                raise ValueError('At least two navigable units are needed for move/next timing.')
            chosen = ids[0]
            # Preparation is outside measurement; force a real group change on every repeat.
            supervisor.label('group', 'unsorted', [chosen])
            supervisor.select([chosen])
            ready()
            row = measure(f'move_best_to_good-{repeat}', actions.move_best_to_good)
            if supervisor.cluster_meta.get('group', chosen) != 'good':
                raise AssertionError('Move did not change the selected unit to good.')
            if row['selected_after'] == row['selected_before']:
                raise AssertionError('Move did not advance selection to the next unit.')
            measure(f'label-{repeat}', lambda: actions.label('group', 'mua'))
            if args.merge_undo:
                supervisor.select(supervisor.cluster_view.get_navigable_ids()[:2])
                ready()
                measure(f'merge-{repeat}', actions.merge)
                measure(f'undo-{repeat}', actions.undo)
        if args.profile:
            ids = supervisor.cluster_view.get_navigable_ids()
            supervisor.label('group', 'unsorted', [ids[0]])
            supervisor.select([ids[0]])
            ready()
            measure('move_best_to_good-profile', actions.move_best_to_good, profiled=True)
            measure('label-profile', lambda: actions.label('group', 'good'), profiled=True)
            if args.merge_undo:
                supervisor.select(supervisor.cluster_view.get_navigable_ids()[:2])
                ready()
                measure('merge-profile', actions.merge, profiled=True)
                measure('undo-profile', actions.undo, profiled=True)
        heartbeat.stop()
        report['status'] = 'ok'
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        cleanup_errors = []
        for obj in (gui, controller):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    cleanup_errors.append(traceback.format_exc())
        if cleanup_errors:
            report['cleanup_errors'] = cleanup_errors
            report['status'] = 'failed'
        report['worker_wall_seconds'] = time.perf_counter() - wall
        report['worker_cpu_seconds'] = time.process_time() - cpu
        from tools.benchmark_long_recording import _peak_rss_bytes

        report['peak_rss_bytes'], report['rss_method'] = _peak_rss_bytes()
        (output / 'worker-result.json').write_text(json.dumps(report, indent=2) + '\n')
        if cleanup_errors:
            raise RuntimeError('GUI cleanup failed; see worker-result.json.')


def run(args):
    started, cpu = time.perf_counter(), time.process_time()
    source, output = _validate(args)
    files, size = _inventory(source, args.max_copy_bytes)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    before = _hashes(source, files)
    report = {
        'source': str(source),
        'output': str(output),
        'source_bytes': size,
        'source_hashes_before': before,
        'status': 'failed',
        'revision': subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        'cache_conditions': 'uncontrolled OS caches; copied bounded package',
        'private_config': str(output / 'config'),
        'configuration': {
            'repeats': args.repeats,
            'events': args.events,
            'merges': args.merges,
            'merge_undo': args.merge_undo,
            'profile': args.profile,
            'max_copy_bytes': args.max_copy_bytes,
            'worker_timeout_seconds': args.timeout,
        },
    }
    server = None
    try:
        from phy.apps.large_recording.file_copy import copy_source_file

        package = output / 'working-package'
        package.mkdir()
        copied = [
            relative
            for relative in files
            if '.phy' not in relative.parts and relative.name != 'phy.log'
        ]
        report['omitted_local_cache_files'] = len(files) - len(copied)
        for relative in copied:
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_source_file(source / relative, target)
        if _hashes(package, copied) != {str(p): before[str(p)] for p in copied}:
            raise RuntimeError('Working copy does not match source hashes.')
        report['copy_and_hash_seconds'] = time.perf_counter() - started
        with (output / 'xvfb.log').open('w') as server_log:
            server = subprocess.Popen(
                [
                    'Xvfb',
                    '-displayfd',
                    '1',
                    '-screen',
                    '0',
                    '1280x900x24',
                    '-nolisten',
                    'tcp',
                    '-ac',
                    '+extension',
                    'GLX',
                    '+render',
                ],
                stdout=subprocess.PIPE,
                stderr=server_log,
                text=True,
            )
            if not select.select([server.stdout], [], [], 20)[0]:
                raise RuntimeError('Timed out starting private Xvfb.')
            display = ':' + server.stdout.readline().strip()
            if display == ':':
                raise RuntimeError('Private Xvfb failed to start.')
            env = dict(
                os.environ,
                DISPLAY=display,
                QT_QPA_PLATFORM='xcb',
                LIBGL_ALWAYS_SOFTWARE='1',
                PYOPENGL_PLATFORM='glx',
                LP_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='2',
                OMP_NUM_THREADS='2',
                XDG_CONFIG_HOME=str(output / 'xdg-config'),
                XDG_CACHE_HOME=str(output / 'xdg-cache'),
                IPYTHONDIR=str(output / 'ipython'),
                JUPYTER_CONFIG_DIR=str(output / 'jupyter'),
                MPLCONFIGDIR=str(output / 'matplotlib'),
            )
            report['private_display'] = display
            report['rendering_environment'] = {
                key: env[key]
                for key in (
                    'QT_QPA_PLATFORM',
                    'LIBGL_ALWAYS_SOFTWARE',
                    'PYOPENGL_PLATFORM',
                    'LP_NUM_THREADS',
                )
            }
            renderer = subprocess.run(
                ['glxinfo', '-B'], env=env, capture_output=True, text=True, timeout=20
            )
            report['renderer_exit_code'] = renderer.returncode
            (output / 'renderer.txt').write_text(renderer.stdout + renderer.stderr)
            renderer.check_returncode()
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                '--worker',
                '--bounded',
                str(source),
                '--output',
                str(output),
                '--repeats',
                str(args.repeats),
                '--merges',
                str(args.merges),
            ]
            if args.events is not None:
                command += ['--events', str(args.events)]
            if args.profile:
                command += ['--profile']
            if args.merge_undo:
                command += ['--merge-undo']
            if args.no_threading:
                command += ['--no-threading']
            with (
                (output / 'worker.stdout').open('w') as stdout,
                (output / 'worker.stderr').open('w') as stderr,
            ):
                try:
                    worker = subprocess.run(
                        command,
                        env=env,
                        cwd=ROOT,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=args.timeout,
                    )
                    report['worker_exit_code'] = worker.returncode
                except subprocess.TimeoutExpired:
                    report['worker_exit_code'] = None
                    report['timed_out'] = True
                    raise
            path = output / 'worker-result.json'
            if path.exists():
                report['worker'] = json.loads(path.read_text())
            worker.check_returncode()
            report['status'] = 'ok'
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            report['xvfb_exit_code'] = server.returncode
        try:
            after_files, _ = _inventory(source, args.max_copy_bytes)
            report['source_hashes_after'] = _hashes(source, after_files)
            report['source_unchanged'] = report['source_hashes_after'] == before
        except Exception:
            report['source_unchanged'] = False
            report['source_hash_error'] = traceback.format_exc()
        report['total_wall_seconds'] = time.perf_counter() - started
        report['parent_cpu_seconds'] = time.process_time() - cpu
        children = os.times()
        report['child_cpu_seconds'] = children.children_user + children.children_system
        report['total_cpu_seconds'] = report['parent_cpu_seconds'] + report['child_cpu_seconds']
        report['peak_child_rss_bytes'] = report.get('worker', {}).get('peak_rss_bytes')
        if not report['source_unchanged']:
            report['status'] = 'failed'
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        if not report['source_unchanged']:
            raise RuntimeError('Input hashes changed during profiling.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bounded', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--events', type=int, help='Synthetic events to append to copy only.')
    parser.add_argument('--merges', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--merge-undo', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--no-threading', action='store_true')
    parser.add_argument('--max-copy-bytes', type=int, default=4 * 1024**3)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.worker:
        _worker(args)
    else:
        report = run(args)
        print(json.dumps({'status': report['status'], 'report': str(args.output / 'result.json')}))


if __name__ == '__main__':
    main()
