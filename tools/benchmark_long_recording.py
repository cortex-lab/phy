"""Reproducible, file-backed capacity benchmark for long-recording services.

Run from a development environment, for example::

    uv run python tools/benchmark_long_recording.py --spikes 1000000

The benchmark creates synthetic Phy arrays without raw traces, times preparation
and replay in separate child processes, and checks replay against an independent
streamed oracle. It is intentionally capped at two million source spikes.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def _file_bytes(path):
    path = Path(path)
    return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())


def _make_source(path, spikes, units, duration_hours, batch_size):
    """Write canonical arrays in bounded chunks, without allocating full vectors."""
    path.mkdir(parents=True)
    duration_seconds = duration_hours * 3600
    sample_rate = 30_000
    num_samples = round(duration_seconds * sample_rate)
    if num_samples <= 0 or num_samples > np.iinfo(np.int64).max:
        raise ValueError('duration-hours produces an invalid int64 sample axis.')

    arrays = {
        'spike_times.npy': np.lib.format.open_memmap(
            path / 'spike_times.npy', mode='w+', dtype=np.int64, shape=(spikes,)
        ),
        'spike_clusters.npy': np.lib.format.open_memmap(
            path / 'spike_clusters.npy', mode='w+', dtype=np.int32, shape=(spikes,)
        ),
        'spike_templates.npy': np.lib.format.open_memmap(
            path / 'spike_templates.npy', mode='w+', dtype=np.int32, shape=(spikes,)
        ),
    }
    for start in range(0, spikes, batch_size):
        stop = min(start + batch_size, spikes)
        rows = np.arange(start, stop, dtype=np.int64)
        arrays['spike_times.npy'][start:stop] = rows * (num_samples - 1) // spikes
        assignments = (rows % units).astype(np.int32)
        arrays['spike_clusters.npy'][start:stop] = assignments
        arrays['spike_templates.npy'][start:stop] = assignments
    for array in arrays.values():
        array.flush()
        array._mmap.close()

    templates = np.lib.format.open_memmap(
        path / 'templates.npy', mode='w+', dtype=np.float32, shape=(units, 4, 2)
    )
    templates[:] = 0
    templates.flush()
    templates._mmap.close()
    np.save(path / 'channel_map.npy', np.array([0, 1], dtype=np.int32))
    np.save(path / 'channel_positions.npy', np.array([[0.0, 0.0], [20.0, 0.0]], dtype=np.float32))
    (path / 'params.py').write_text(
        f'sample_rate = {sample_rate}.0\n'
        f'num_samples = {num_samples}\n'
        'n_channels_dat = 2\n'
        "dtype = 'int16'\n"
        'dat_path = None\n',
        encoding='utf-8',
    )
    return duration_seconds


def _peak_rss_bytes():
    """Return process peak RSS, or an explicit unavailable result."""
    if os.name != 'nt':
        try:
            import resource

            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux and the BSDs report KiB; macOS reports bytes.
            return int(value if sys.platform == 'darwin' else value * 1024), 'resource.ru_maxrss'
        except (ImportError, AttributeError, OSError):
            pass

    try:
        import psutil
    except ImportError:
        return None, 'unavailable (install psutil for sampled peak RSS on Windows)'

    process = psutil.Process()
    peak = [process.memory_info().rss]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.02):
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    return _SampledRSS(stopped, sampler, peak)


class _SampledRSS:
    def __init__(self, stopped, sampler, peak):
        self.stopped = stopped
        self.sampler = sampler
        self.peak = peak

    def finish(self):
        self.stopped.set()
        self.sampler.join()
        return int(self.peak[0]), 'psutil sampled every 20 ms'


def _measure_rss_start():
    value = _peak_rss_bytes()
    return value if isinstance(value, _SampledRSS) else value


def _rss_result(handle):
    return handle.finish() if isinstance(handle, _SampledRSS) else handle


def _phase_prepare(args):
    from phy.apps.large_recording.prepare import prepare_large

    rss = _measure_rss_start()
    started = time.perf_counter()
    bounded = prepare_large(
        args.params,
        args.bounded_path,
        duration_seconds=args.duration_seconds,
        max_spikes=args.max_spikes,
        max_bytes=args.max_bytes,
        window_seconds=args.window_seconds,
        stratum_seconds=args.stratum_seconds,
        seed=args.seed,
        batch_size=args.batch_size,
        bin_seconds=args.bin_seconds,
    )
    elapsed = time.perf_counter() - started
    peak, method = _rss_result(rss)
    print(
        json.dumps(
            {
                'phase': 'prepare',
                'wall_seconds': elapsed,
                'peak_rss_bytes': peak,
                'peak_rss_method': method,
                'bounded_path': str(bounded),
            }
        )
    )


def _phase_curate(args):
    from phy.apps.large_recording.curation import BoundedSession

    session = BoundedSession(args.bounded_path)
    try:
        if 0 not in session.unit_ids or 1 not in session.unit_ids:
            raise ValueError('Synthetic source must contain units 0 and 1.')
        session.record_merge([0, 1], session.next_cluster_id)
        session.save()
    finally:
        session.close()
    print(json.dumps({'phase': 'curate', 'saved': True}))


def _phase_replay(args):
    from phy.apps.large_recording.replay import apply_curation

    rss = _measure_rss_start()
    started = time.perf_counter()
    output = apply_curation(
        args.params, args.bounded_path, args.curated_path, batch_size=args.batch_size
    )
    elapsed = time.perf_counter() - started
    peak, method = _rss_result(rss)
    print(
        json.dumps(
            {
                'phase': 'replay',
                'wall_seconds': elapsed,
                'peak_rss_bytes': peak,
                'peak_rss_method': method,
                'curated_path': str(output),
            }
        )
    )


def _oracle(source_path, curated_path, spikes, units, batch_size):
    """Independent streamed oracle for the benchmark's one known merge."""
    source = np.load(source_path / 'spike_clusters.npy', mmap_mode='r', allow_pickle=False)
    output = np.load(curated_path / 'spike_clusters.npy', mmap_mode='r', allow_pickle=False)
    expected_target = units
    checked = 0
    matches = source.shape == output.shape == (spikes,)
    for start in range(0, spikes, batch_size):
        stop = min(start + batch_size, spikes)
        expected = np.array(source[start:stop], copy=True)
        expected[(expected == 0) | (expected == 1)] = expected_target
        matches &= np.array_equal(expected, output[start:stop])
        checked += stop - start
    source._mmap.close()
    output._mmap.close()
    return {'matches': bool(matches), 'rows_checked': checked, 'merge_target': expected_target}


def _run_phase(
    script,
    phase,
    params,
    bounded,
    curated,
    duration_seconds,
    max_spikes,
    max_bytes,
    window_seconds,
    stratum_seconds,
    seed,
    batch_size,
    bin_seconds,
):
    command = [
        sys.executable,
        str(script),
        '--_phase',
        phase,
        '--params',
        str(params),
        '--bounded-path',
        str(bounded),
        '--curated-path',
        str(curated),
        '--duration-seconds',
        str(duration_seconds),
        '--max-spikes',
        str(max_spikes),
        '--max-bytes',
        str(max_bytes),
        '--window-seconds',
        str(window_seconds),
        '--stratum-seconds',
        str(stratum_seconds),
        '--seed',
        str(seed),
        '--batch-size',
        str(batch_size),
        '--bin-seconds',
        str(bin_seconds),
    ]
    try:
        completed = subprocess.run(
            command, cwd=REPOSITORY, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.strip() or error.stdout.strip()) from error
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _run(args):
    if args.spikes <= 0 or args.spikes > 2_000_000:
        raise ValueError('--spikes must be between 1 and 2,000,000.')
    if args.units < 2 or args.units > args.spikes:
        raise ValueError('--units must be at least 2 and no greater than --spikes.')
    if args.batch_size <= 0 or args.duration_hours <= 0:
        raise ValueError('--batch-size and --duration-hours must be positive.')

    temporary = None
    if args.output_dir:
        root = Path(args.output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix='phy-long-recording-benchmark-')
        root = Path(temporary.name)
    source_dir, bounded_dir, curated_dir = (
        root / name for name in ('source', 'bounded', 'curated')
    )
    duration_seconds = _make_source(
        source_dir, args.spikes, args.units, args.duration_hours, args.batch_size
    )
    source_bytes = _file_bytes(source_dir)
    common = {
        'params': source_dir / 'params.py',
        'bounded': bounded_dir,
        'curated': curated_dir,
        'duration_seconds': duration_seconds,
        'max_spikes': args.max_spikes,
        'max_bytes': args.max_bytes,
        'window_seconds': args.window_seconds,
        'stratum_seconds': args.stratum_seconds,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'bin_seconds': args.bin_seconds,
    }
    prepare = _run_phase(Path(__file__).resolve(), 'prepare', **common)
    _run_phase(Path(__file__).resolve(), 'curate', **common)
    replay = _run_phase(Path(__file__).resolve(), 'replay', **common)
    oracle = _oracle(source_dir, curated_dir, args.spikes, args.units, args.batch_size)
    if not oracle['matches']:
        raise RuntimeError('Independent streamed replay oracle did not match curated assignments.')

    result = {
        'benchmark': 'phy_long_recording_v1',
        'machine': {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'cpu_count': os.cpu_count(),
        },
        'parameters': {
            'spikes': args.spikes,
            'units': args.units,
            'duration_hours': args.duration_hours,
            'sample_rate_hz': 30000,
            'batch_size': args.batch_size,
            'max_spikes': args.max_spikes,
            'max_bytes': args.max_bytes,
            'window_seconds': args.window_seconds,
            'stratum_seconds': args.stratum_seconds,
            'seed': args.seed,
            'bin_seconds': args.bin_seconds,
        },
        'source_file_bytes': source_bytes,
        'bounded_file_bytes': _file_bytes(bounded_dir),
        'curated_file_bytes': _file_bytes(curated_dir),
        'bounded_spikes': int(
            json.loads((bounded_dir / '_phy_curation_manifest.json').read_text(encoding='utf-8'))[
                'bounded_num_spikes'
            ]
        ),
        'preparation': prepare,
        'replay': replay,
        'independent_streamed_replay_oracle': oracle,
        'output_dir': str(root),
        'output_dir_retained': temporary is None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if temporary is not None:
        temporary.cleanup()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spikes', type=int, default=1_000_000)
    parser.add_argument('--units', type=int, default=1000)
    parser.add_argument('--duration-hours', type=float, default=1000)
    parser.add_argument('--batch-size', type=int, default=100_000)
    parser.add_argument('--max-spikes', type=int, default=100_000)
    parser.add_argument('--max-bytes', type=int, default=536_870_912)
    parser.add_argument('--window-seconds', type=float, default=30)
    parser.add_argument('--stratum-seconds', type=float, default=3600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bin-seconds', type=float, default=3600)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument(
        '--_phase', choices=('prepare', 'curate', 'replay'), help=argparse.SUPPRESS
    )
    parser.add_argument('--params', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--bounded-path', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--curated-path', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--duration-seconds', type=float, help=argparse.SUPPRESS)
    return parser


def main():
    args = _parser().parse_args()
    if args._phase == 'prepare':
        _phase_prepare(args)
    elif args._phase == 'curate':
        _phase_curate(args)
    elif args._phase == 'replay':
        _phase_replay(args)
    else:
        _run(args)


if __name__ == '__main__':
    main()
