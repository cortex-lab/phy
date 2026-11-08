"""Run bounded ordinary-application observations in a disposable container."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('release', 'candidate', 'both'), default='both')
    parser.add_argument('--scale', type=int, choices=(1, 2), default=2)
    parser.add_argument('--sommelier', action='store_true')
    parser.add_argument('--inject-worker-error', action='store_true')
    parser.add_argument('--without-raw-data', action='store_true')
    args = parser.parse_args()
    if not Path('/data/params.py').is_file():
        parser.error('Mount a disposable/public test dataset at /data read-only')
    output = Path(tempfile.mkdtemp(prefix='wayland-', dir='/work'))
    runtime = output / 'runtime'
    runtime.mkdir(mode=0o700)
    dataset = output / 'dataset'
    shutil.copytree('/data', dataset, ignore=shutil.ignore_patterns('.phy', '*.log'))
    if args.without_raw_data:
        for pattern in ('*.dat', '*.bin'):
            for recording in dataset.glob(pattern):
                recording.unlink()  # Only the disposable copy, never /data.
    env = dict(
        os.environ,
        XDG_RUNTIME_DIR=str(runtime),
        WAYLAND_DISPLAY='wayland-phy',
        XDG_SESSION_TYPE='wayland',
        QT_QPA_PLATFORM='wayland',
        DISPLAY=':0',
        LIBGL_ALWAYS_SOFTWARE='true',
        GALLIUM_DRIVER='llvmpipe',
        LP_NATIVE_VECTOR_WIDTH='128',
    )
    for key in ('PYOPENGL_PLATFORM', 'QT_XCB_GL_INTEGRATION', 'PYTHONPATH'):
        env.pop(key, None)
    command = [
        'weston',
        '--backend=headless',
        '--renderer=gl',
        '--no-config',
        '--socket=wayland-phy',
        '--width=2800',
        '--height=1800',
        '--scale=' + str(args.scale),
        '--idle-time=0',
        '--xwayland',
        '--log=' + str(output / 'weston.log'),
    ]
    print('Evidence:', output, flush=True)
    summaries = {}
    with (output / 'weston.stdout').open('w') as log:
        compositor = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 15
            while not (runtime / 'wayland-phy').exists():
                if compositor.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Weston did not start; inspect weston.log')
                time.sleep(0.05)
            modes = ('release', 'candidate') if args.source == 'both' else (args.source,)
            for mode in modes:
                child_env = dict(env)
                if mode == 'candidate':
                    child_env['PYTHONPATH'] = '/src'
                probe = [
                    '/opt/phy210/bin/python',
                    '/src/scripts/tests/wayland/application.py',
                    str(dataset / 'params.py'),
                    '--trace',
                    str(output / (mode + '.json')),
                    '--log',
                    str(output / (mode + '.log')),
                ]
                if args.inject_worker_error:
                    probe.append('--inject-worker-error')
                if args.sommelier:
                    probe = [
                        '/opt/sommelier-src/build/sommelier',
                        '--noop-driver',
                        '--display=wayland-phy',
                        '--scale=1',
                        *probe,
                    ]
                with (output / (mode + '.stdout')).open('w') as stdout:
                    process = subprocess.Popen(
                        probe,
                        env=child_env,
                        stdout=stdout,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        summaries[mode] = {'returncode': process.wait(timeout=40)}
                    except subprocess.TimeoutExpired:
                        summaries[mode] = {'returncode': None, 'error': 'process timed out'}
                    finally:
                        # Sommelier may launch a child. Do not let a stuck child
                        # contaminate the next source comparison.
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                trace = output / (mode + '.json')
                if trace.exists():
                    summaries[mode]['status'] = json.loads(trace.read_text()).get('status')
                print(mode, summaries[mode], flush=True)
        finally:
            compositor.terminate()
            try:
                compositor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                compositor.kill()
                compositor.wait()
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(compositor.pid, signal.SIGKILL)
    (output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    return (
        0
        if summaries
        and all(
            value.get('returncode') == 0 and value.get('status') == 'PASS'
            for value in summaries.values()
        )
        else 1
    )


if __name__ == '__main__':
    raise SystemExit(main())
