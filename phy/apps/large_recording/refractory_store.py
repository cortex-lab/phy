"""Optional exact RP artifacts, bound to an authenticated prepared snapshot."""

import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np

from .live_summaries import LiveSummaryEngine, _check_cancel, _signature
from .refractory import (
    BIN_SECONDS,
    VERSION,
    RefractoryAccumulator,
    RefractoryGeometry,
    RefractoryUnavailable,
    window_curve,
)
from .refractory_resources import (
    DEFAULT_RP_MAX_BYTES,
    DEFAULT_RP_MEMORY_BYTES,
    estimate_refractory_disk,
    plan_refractory_summaries,
)
from .source import hash_file, hash_json, publish_directory
from .summary_store import (
    StoreIntegrityError,
    SummaryStore,
    _atomic_json,
    _attachment_lock,
    summary_store_reference,
)

REFERENCE = 'refractory-summaries.json'


def refractory_binding(source):
    """Include scientific parameters as well as source identity and assignments."""
    m = source.manifest
    return {
        'algorithm': VERSION,
        'bin_seconds': BIN_SECONDS,
        'identity': m['source']['identity'],
        'revision': m['source']['revision'],
        'sample_rate': m['sample_rate'],
        'num_samples': m['num_samples'],
        'valid_intervals': m['valid_intervals'],
    }


def estimate_refractory_summaries(source):
    """Conservative disk bound, including per-unit array headers and metadata."""
    m = source.manifest
    samples = round(BIN_SECONDS * m['sample_rate'])
    if samples <= 0:
        raise RefractoryUnavailable('Unsupported RP sample rate.')
    bins = (m['num_samples'] + samples - 1) // samples
    return estimate_refractory_disk(m, bins)


def _prepare_counters(source, plan, geometry, cancel, progress):
    m = source.manifest
    names = {'spike_times.npy', 'spike_clusters.npy', 'amplitudes.npy'}
    if 'amplitudes.npy' not in m['source_array_inventory']:
        raise RefractoryUnavailable('Raw source amplitudes unavailable.')
    source._open(names, cancel, progress)
    accumulators = {
        u: RefractoryAccumulator(
            m['sample_rate'],
            m['num_samples'],
            m['valid_intervals'],
            max_component=plan.max_component,
            geometry=geometry,
        )
        for u in m['unit_ids']
    }
    unavailable = {}
    chunk = plan.batch_size
    for start in range(0, m['num_spikes'], chunk):
        _check_cancel(cancel)
        stop = min(start + chunk, m['num_spikes'])
        ticks, labels, amplitudes = (
            source.arrays[n][start:stop].reshape(-1)
            for n in ('spike_times.npy', 'spike_clusters.npy', 'amplitudes.npy')
        )
        # Group each source batch once, instead of scanning it once per unit.
        order = np.argsort(labels, kind='stable')
        sorted_labels = labels[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1, len(order)]
        for first, last in zip(boundaries[:-1], boundaries[1:]):
            _check_cancel(cancel)
            unit = int(sorted_labels[first])
            if unit in unavailable:
                continue
            selected = order[first:last]
            try:
                accumulators[unit].feed(ticks[selected], amplitudes[selected], cancel)
            except RefractoryUnavailable as error:
                unavailable[unit] = {'status': 'unavailable', 'message': str(error)}
                del accumulators[unit]
        if progress:
            progress('preparing RP summaries', stop, m['num_spikes'])
    source._unchanged()
    for unit in m['unit_ids']:
        _check_cancel(cancel)
        if unit in unavailable:
            yield unit, unavailable.pop(unit)
        else:
            acc = accumulators.pop(unit)
            yield unit, acc.finish_counters(cancel)


def build_refractory_summaries(
    package_path,
    source_dir=None,
    max_bytes=DEFAULT_RP_MAX_BYTES,
    memory_bytes=DEFAULT_RP_MEMORY_BYTES,
    batch_size=100000,
    cancel=None,
    progress=None,
):
    """Publish original-unit counters atomically, independently of row stores.

    A single authenticated source scan maintains complete close components for
    every original unit. One plan covers counters, shared exposure, unresolved
    components, source batches and serialization. No display curves are built.
    Existing artifacts are never overwritten. The package survives failure.
    """
    output = Path(package_path) / '.phy' / 'refractory-summaries'
    if output.exists():
        raise ValueError(f'RP summaries already exist: {output}.')
    if min(max_bytes, memory_bytes, batch_size) <= 0:
        raise ValueError('RP budgets must be positive.')
    with LiveSummaryEngine(
        package_path,
        source_dir=source_dir,
        batch_size=batch_size,
        max_bytes=memory_bytes,
    ) as source:
        m = source.manifest
        plan = plan_refractory_summaries(
            m,
            max_bytes,
            memory_bytes,
            batch_size,
            native_grid_bytes=source.edges.nbytes + source.exposure.nbytes,
        )
        geometry = RefractoryGeometry(m['sample_rate'], m['num_samples'], m['valid_intervals'])
        if progress:
            progress(
                f'RP plan: {plan.units} units × {plan.bins} bins; '
                f'disk <= {plan.disk_bytes} bytes; allocations <= {plan.memory_bytes} bytes; '
                f'batch {plan.batch_size}',
                0,
                m['num_spikes'],
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < plan.disk_bytes:
            raise OSError(f'RP summaries need {plan.disk_bytes} bytes of free disk space.')
        temporary = Path(tempfile.mkdtemp(prefix='.rp-building-', dir=output.parent))
        try:
            results = _prepare_counters(source, plan, geometry, cancel, progress)
            catalog = {'binding': refractory_binding(source), 'units': {}}
            for index, (unit, result) in enumerate(results):
                _check_cancel(cancel)
                arrays = {k: v for k, v in result.items() if k in ('spikes', 'violations')}
                item = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
                if arrays:
                    filename = f'{index}.npz'
                    with (temporary / filename).open('wb') as stream:
                        np.savez(stream, **arrays)
                        stream.flush()
                        os.fsync(stream.fileno())
                    item.update(file=filename, sha256=hash_file(temporary / filename))
                catalog['units'][str(unit)] = item
            _atomic_json(temporary / 'index.json', catalog)
            if sum(p.stat().st_size for p in temporary.iterdir()) > max_bytes:
                raise ValueError('RP summaries exceed disk budget.')
            _check_cancel(cancel)
            lock = output.parent / 'rp-summary-publication'
            lock.mkdir(exist_ok=True)
            with _attachment_lock(lock, lambda: _check_cancel(cancel)):
                if output.exists():
                    raise ValueError(f'RP summaries already exist: {output}.')
                # Pin before publication: interruption can leave a dangling
                # reference, but never an unattachable completed directory.
                _atomic_json(
                    output.parent / REFERENCE,
                    {'index_sha256': hash_json(catalog)},
                )
                publish_directory(temporary, output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return output


def _counters(source, expressions, cancel, result_bytes=8388608):
    path = source.path / '.phy'
    reference = path / REFERENCE
    if not reference.exists():
        return {}
    catalog_path = path / 'refractory-summaries' / 'index.json'
    if reference.stat().st_size > 65536 or catalog_path.stat().st_size > 16777216:
        raise StoreIntegrityError('RP catalog exceeds limit.')
    reference_bytes = reference.read_bytes()
    catalog = json.loads(catalog_path.read_bytes())
    if hash_json(catalog) != json.loads(reference_bytes)['index_sha256'] or catalog[
        'binding'
    ] != refractory_binding(source):
        raise StoreIntegrityError('RP summary binding mismatch.')
    originals = {u: e for u, e in expressions.items() if e['kind'] == 'source'}
    if _result_bytes(source, len(originals)) > result_bytes:
        return {
            u: {'status': 'unavailable', 'message': 'RP result grids exceed memory budget.'}
            for u in originals
        }
    results = {}
    recorded = None
    for unit, expression in expressions.items():
        _check_cancel(cancel)
        if expression['kind'] != 'source':
            continue
        item = catalog['units'][str(expression['unit'])]
        result = {k: v for k, v in item.items() if k not in ('file', 'sha256')}
        if 'file' in item:
            filename = item['file']
            if Path(filename).name != filename:
                raise StoreIntegrityError('Invalid RP array path.')
            with (catalog_path.parent / filename).open('rb') as stream:
                raw = stream.read(16777217)
                if len(raw) > 16777216:
                    raise StoreIntegrityError('RP unit arrays exceed limit.')
                if hashlib.sha256(raw).hexdigest() != item['sha256']:
                    raise StoreIntegrityError('RP summary checksum mismatch.')
                with np.load(BytesIO(raw), allow_pickle=False) as data:
                    result.update({k: data[k] for k in data.files})
            if recorded is None:
                m = source.manifest
                recorded = RefractoryGeometry(
                    m['sample_rate'], m['num_samples'], m['valid_intervals']
                ).recorded
            centres, curve = window_curve(
                result['violations'], result['spikes'], recorded, source.manifest['sample_rate']
            )
            result.update(recorded_seconds=recorded, window_centres_h=centres, contamination=curve)
        for value in result.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        results[unit] = result
    _check_cancel(cancel)
    if reference.read_bytes() != reference_bytes:
        raise StoreIntegrityError('RP summary reference changed while reading.')
    return results


def _unit_batches(store, source, root, batch_size, cancel):
    previous = -1
    for rows in store.batches(root, batch_size, cancel):
        bins = rows['bin'].astype(np.int64)
        if np.any(bins < 0) or np.any(bins >= source.bins):
            raise StoreIntegrityError('Timestamp bin outside native grid.')
        offsets = rows['tick_offset']
        if np.any(offsets >= (source.edges[bins + 1] - source.edges[bins]).astype(np.uint64)):
            raise StoreIntegrityError('Timestamp offset outside native bin.')
        ticks = source.edges[bins] + offsets.astype(np.int64)
        if len(ticks) and (ticks[0] < previous or np.any(np.diff(ticks) < 0)):
            raise StoreIntegrityError('Timestamp rows must be monotonic.')
        if len(ticks):
            previous = int(ticks[-1])
        yield {
            'spike_times.npy': ticks,
            'spike_clusters.npy': np.full(len(rows), root, dtype=np.int64),
            'amplitudes.npy': rows['amplitude'],
        }


def _chronological(store, source, roots, cancel):
    """Merge ordered unit chunks using watermarks; scratch never scales with spikes."""
    roots = sorted(roots)
    # Account for buffers, packed arrays, masks, sort indices and feed scratch.
    if len(roots) * 512 > source.max_bytes:
        raise RefractoryUnavailable('RP ancestor streams exceed memory budget.')
    batch_size = min(source.batch_size, max(1, source.max_bytes // (512 * len(roots))))
    streams = [_unit_batches(store, source, r, batch_size, cancel) for r in roots]
    active = {i: next(stream, None) for i, stream in enumerate(streams)}
    active = {i: b for i, b in active.items() if b is not None}
    while active:
        _check_cancel(cancel)
        watermark = min(b['spike_times.npy'][-1] for b in active.values())
        parts = []
        for i, batch in list(active.items()):
            stop = np.searchsorted(batch['spike_times.npy'], watermark, side='right')
            if stop:
                parts.append({k: v[:stop] for k, v in batch.items()})
            if stop == len(batch['spike_times.npy']):
                following = next(streams[i], None)
                if following is None:
                    del active[i]
                else:
                    active[i] = following
            else:
                active[i] = {k: v[stop:] for k, v in batch.items()}
        batch = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.argsort(batch['spike_times.npy'], kind='stable')
        yield {k: v[order] for k, v in batch.items()}


def _result_bytes(source, units):
    m = source.manifest
    samples = round(BIN_SECONDS * m['sample_rate'])
    if samples <= 0:
        raise RefractoryUnavailable('Unsupported RP sample rate.')
    bins = (m['num_samples'] + samples - 1) // samples
    return units * (4096 + bins * 40)


def resolve_prepared(engine, expressions, cancel, progress):
    """Try verified snapshot evidence before any source open/stat/hash operation."""
    from .refractory_engine import ancestors

    source = engine.source
    results = {}
    try:
        results = _counters(source, expressions, cancel, engine.result_bytes)
        if results:
            engine.last_path = 'prepared RP counters'
    except (OSError, ValueError, KeyError, TypeError):
        pass  # Authenticated source fallback remains available.
    remaining = {u: e for u, e in expressions.items() if u not in results}
    supported = {
        u: e for u, e in remaining.items() if not source._required(e) - {'amplitudes.npy'}
    }
    if not supported or summary_store_reference(source.path, allow_invalid=True) is None:
        return results
    try:
        reference = summary_store_reference(source.path)
        with SummaryStore(source) as store:
            if not store.timestamps or 'amplitude' not in store.dtype.names:
                return results
            if (
                _result_bytes(source, len(supported))
                + sum(engine._bytes(value) for value in results.values())
                > engine.result_bytes
            ):
                results.update(
                    {
                        u: {
                            'status': 'unavailable',
                            'message': 'RP result grids exceed memory budget.',
                        }
                        for u in supported
                    }
                )
                return results
            stamp = (
                reference,
                hash_json(store.index),
                _signature(store.signature),
                _signature(store.path_signature),
            )
            if stamp != engine.artifact_stamp:
                engine.prepared_results = OrderedDict()
                engine.artifact_stamp = stamp
            pending = {}
            for unit, expression in supported.items():
                _check_cancel(cancel)
                key = engine.key(expression)
                if key in engine.prepared_results:
                    pending[unit] = engine.prepared_results[key]
                    continue
                m = source.manifest
                acc = RefractoryAccumulator(
                    m['sample_rate'], m['num_samples'], m['valid_intervals']
                )
                error = None
                for batch in _chronological(store, source, ancestors(expression), cancel):
                    if error is None:
                        try:
                            mask = source._mask(expression, batch)
                            acc.feed(
                                batch['spike_times.npy'][mask],
                                batch['amplitudes.npy'][mask],
                                cancel,
                            )
                        except RefractoryUnavailable as exc:
                            error = {'status': 'unavailable', 'message': str(exc)}
                    if progress:
                        progress('RP prepared timestamp rows', 0, None)
                pending[unit] = error if error else acc.finish(cancel)
            _check_cancel(cancel)
            # Recheck attachment and signatures even for a cached result.
            if (
                summary_store_reference(source.path) != stamp[0]
                or _signature(os.fstat(store.stream.fileno())) != stamp[2]
                or _signature((store.path / 'data.bin').stat()) != stamp[3]
            ):
                raise StoreIntegrityError('RP timestamp snapshot changed while reading.')
            for unit, value in pending.items():
                engine.prepared_results[engine.key(supported[unit])] = value
            engine._trim(engine.prepared_results, engine.result_bytes)
            results.update(pending)
            engine.last_path = 'prepared timestamp rows'
    except (OSError, ValueError, KeyError, TypeError):
        engine.artifact_stamp = None
    return results
