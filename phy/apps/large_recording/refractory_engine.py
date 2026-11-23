"""Worker-only demand scans and bounded timestamp-bearing acceleration."""

from collections import OrderedDict

import numpy as np

from .live_summaries import _check_cancel
from .refractory import VERSION, RefractoryAccumulator, RefractoryUnavailable
from .source import hash_json


def ancestors(expression):
    if expression['kind'] == 'source':
        return {expression['unit']}
    if expression['kind'] == 'split':
        return ancestors(expression['parent'])
    return set().union(*(ancestors(p) for p in expression['parts']))


class RefractoryEngine:
    """Non-additive results; cached root rows retain original signed precision.

    The source lease is serialized with other recording I/O. Each reuse checks
    the pinned reader signatures; only full content authentication creates trust.
    Root rows are disposable and never alter the sorting or export membership.
    """

    def __init__(self, source, root_bytes=268435456, result_bytes=8388608):
        self.source = source
        self.root_bytes, self.result_bytes = root_bytes, result_bytes
        self.roots, self.results = OrderedDict(), OrderedDict()
        self.last_path = None
        m = source.manifest
        self.binding = hash_json(
            {
                'version': VERSION,
                'identity': m['source']['identity'],
                'revision': m['source']['revision'],
                'fs': m['sample_rate'],
                'num_samples': m['num_samples'],
                'intervals': m['valid_intervals'],
            }
        )

    def key(self, expression):
        return hash_json({'binding': self.binding, 'membership': expression})

    @staticmethod
    def _bytes(value):
        return sum(v.nbytes for v in value.values() if isinstance(v, np.ndarray))

    def _trim(self, cache, budget):
        size = sum(self._bytes(v) for v in cache.values())
        while cache and (size > budget or len(cache) > 512):
            _, value = cache.popitem(last=False)
            size -= self._bytes(value)

    def resolve(self, expressions, cancel=None, progress=None):
        s, m = self.source, self.source.manifest
        names = {'spike_times.npy', 'spike_clusters.npy', 'amplitudes.npy'}
        for e in expressions.values():
            names |= s._required(e)
        if 'amplitudes.npy' not in m['source_array_inventory']:
            raise RefractoryUnavailable('Raw source amplitudes unavailable.')
        s._open(names, cancel, progress)
        results, missing = {}, {}
        for unit, expression in expressions.items():
            key = self.key(expression)
            if key in self.results:
                results[unit] = self.results[key]
                self.results.move_to_end(key)
            else:
                missing[unit] = expression
        if not missing:
            self.last_path = 'result cache'
            return results
        required_roots = set().union(*(ancestors(e) for e in missing.values()))
        accumulators = {
            u: RefractoryAccumulator(
                m['sample_rate'],
                m['num_samples'],
                m['valid_intervals'],
                max_component=max(1, 1000000 // len(missing)),
            )
            for u in missing
        }
        cached = all(
            root in self.roots and names <= self.roots[root].keys() for root in required_roots
        )
        pending_roots, cache_size = {}, 0
        # Admit roots before scanning, using exact manifest counts. Oversized
        # roots never transiently accumulate a recording-sized list of rows.
        counts = dict(zip(m['unit_ids'], m['counts'], strict=True))
        row_bytes = sum(np.dtype(m['source_array_inventory'][n]['dtype']).itemsize for n in names)
        admitted = set()
        for root in sorted(required_roots):
            cost = counts[root] * row_bytes
            if cache_size + cost <= self.root_bytes:
                admitted.add(root)
                cache_size += cost
        cache_size = 0
        if cached:
            self.last_path = 'unit ticks'
            rows = {
                name: np.concatenate([self.roots[r][name] for r in sorted(required_roots)])
                for name in names
            }
            order = np.argsort(rows['spike_times.npy'], kind='stable')
            rows = {n: v[order] for n, v in rows.items()}
            total = len(order)
            for r in required_roots:
                self.roots.move_to_end(r)
        else:
            self.last_path = 'source scan'
            total = m['num_spikes']
            rows = s.arrays
        previous = -1
        unavailable = {}
        for start in range(0, total, s._chunk_size(missing, 128)):
            _check_cancel(cancel)
            stop = min(start + s._chunk_size(missing, 128), total)
            batch = {n: rows[n][start:stop].reshape(-1) for n in names}
            ticks = batch['spike_times.npy']
            if ticks.dtype.kind not in 'iu' or (
                len(ticks) and (ticks.min() < 0 or ticks.max() > np.iinfo(np.int64).max)
            ):
                raise RefractoryUnavailable('Source timestamps exceed signed int64 range.')
            ticks = ticks.astype(np.int64, copy=False)
            batch['spike_times.npy'] = ticks
            if len(ticks) and (
                ticks.dtype.kind not in 'iu'
                or ticks[0] < previous
                or np.any(np.diff(ticks) < 0)
                or ticks[0] < 0
                or ticks[-1] >= m['num_samples']
            ):
                raise RefractoryUnavailable('Source timestamps are not valid monotonic ticks.')
            if len(ticks):
                previous = int(ticks[-1])
            memo = {}
            for unit, expression in missing.items():
                _check_cancel(cancel)
                if unit in unavailable:
                    continue
                try:
                    selected = s._mask(expression, batch, memo)
                    accumulators[unit].feed(
                        ticks[selected], batch['amplitudes.npy'][selected], cancel
                    )
                except ValueError as error:
                    unavailable[unit] = {'status': 'unavailable', 'message': str(error)}
            if not cached and cache_size <= self.root_bytes:
                for root in admitted:
                    selected = s._mask({'kind': 'source', 'unit': root}, batch, memo)
                    part = {n: v[selected].copy() for n, v in batch.items()}
                    cache_size += self._bytes(part)
                    pending_roots.setdefault(root, []).append(part)
                if cache_size > self.root_bytes:
                    pending_roots.clear()
            if progress:
                progress(f'RP {self.last_path}', stop, total)
        _check_cancel(cancel)
        s._unchanged()
        for unit, accumulator in accumulators.items():
            value = unavailable[unit] if unit in unavailable else accumulator.finish(cancel)
            self.results[self.key(missing[unit])] = value
            results[unit] = value
        for root, parts in pending_roots.items():
            self.roots[root] = {n: np.concatenate([p[n] for p in parts]) for n in names}
        self._trim(self.roots, self.root_bytes)
        self._trim(self.results, self.result_bytes)
        return results
