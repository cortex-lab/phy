"""Synthetic cached-file comparison; run from a synced development checkout."""

import json
import platform
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from phy.apps.large_recording.prepare import prepare_large
from phy.apps.large_recording.range_build import build_amplitude_index
from phy.apps.large_recording.source import Source, hash_file, hash_files
from phy.apps.large_recording.summaries import build_summaries
from phy.apps.large_recording.summary_store import build_summary_store


def measured(label, fn):
    tracemalloc.start()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(label, 'seconds', round(elapsed, 4), 'traced_peak_bytes', peak, flush=True)
    return result


print(platform.platform(), platform.processor(), 'numpy', np.__version__, flush=True)
with tempfile.TemporaryDirectory(prefix='phy-bench-') as temporary:
    root = Path(temporary)
    path = root / 'source'
    path.mkdir()
    n = 8_000_000
    rng = np.random.default_rng(0)
    arrays = {
        'spike_times.npy': np.arange(n, dtype=np.int64) * 80000 // n,
        'spike_clusters.npy': rng.integers(0, 16, n, dtype=np.int32),
        'spike_templates.npy': np.zeros(n, np.int32),
        'amplitudes.npy': rng.standard_normal(n).astype(np.float32),
        'spike_depths.npy': rng.standard_normal(n).astype(np.float32),
    }
    for name, values in arrays.items():
        np.save(path / name, values)
    np.save(path / 'templates.npy', np.zeros((1, 3, 1), np.float32))
    np.save(path / 'channel_map.npy', np.array([0], np.int32))
    np.save(path / 'channel_positions.npy', np.array([[0.0, 0.0]]))
    (path / 'params.py').write_text('sample_rate=1000\nnum_samples=80000\ndat_path=None\n')
    paths = {name: path / name for name in arrays}
    hs = []
    for w in (1, 4):
        hs.append(measured('hash ' + str(w), lambda w=w: hash_files(paths, workers=w)))
    assert hs[0] == hs[1]
    bounded = prepare_large(
        path / 'params.py',
        root / 'bounded',
        max_spikes=100000,
        max_bytes=128 * 1024 * 1024,
        window_seconds=0.01,
        stratum_seconds=80,
        bin_seconds=40,
    )
    stores = []
    for w in (1, 4):
        stores.append(
            measured(
                'rows ' + str(w),
                lambda w=w: build_summary_store(
                    bounded, root / ('rows' + str(w)), workers=w, buffer_bytes=256 * 1024 * 1024
                ),
            )
        )
    assert hash_file(stores[0] / 'data.bin') == hash_file(stores[1] / 'data.bin')
    indices = []
    for w in (1, 4):
        indices.append(
            measured(
                'index ' + str(w),
                lambda w=w: build_amplitude_index(
                    bounded,
                    root / ('index' + str(w)),
                    workers=w,
                    attach=False,
                    memory_bytes=256 * 1024 * 1024,
                ),
            )
        )
    for name in ('values.bin', 'nodes.bin', 'segments.bin'):
        assert hash_file(indices[0] / name) == hash_file(indices[1] / name)
    print('ALL OUTPUTS EXACT', flush=True)
