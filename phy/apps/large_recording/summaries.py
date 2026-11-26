"""Exact file-backed counts and additive source-measurement summaries."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .source import close_array
from .streaming import UnitRows, bounded_map, check_cancel, worker_count


def scalar_attributes(source):
    """Return available scalar measurements without inventing physical units."""
    attributes = {}
    amplitude = source.arrays.get('amplitudes.npy')
    if amplitude is not None and (amplitude.ndim == 1 or amplitude.shape[1:] == (1,)):
        attributes['amplitude'] = amplitude.reshape(-1)
    depth = source.arrays.get('spike_depths.npy')
    if depth is not None and (depth.ndim == 1 or depth.shape[1:] == (1,)):
        attributes['depth'] = depth.reshape(-1)
    elif (positions := source.arrays.get('spike_positions.npy')) is not None:
        axis = source.params.get('depth_axis')
        if axis is not None:
            if (
                not isinstance(axis, int)
                or positions.ndim != 2
                or not 0 <= axis < positions.shape[1]
            ):
                raise ValueError('depth_axis must declare a valid spike_positions.npy coordinate.')
            attributes['depth'] = positions[:, axis]
    return attributes


def summary_size(source, bin_seconds=3600):
    """Estimate exact array bytes before allocating the dense output grid."""
    step = float(bin_seconds) * source.sample_rate
    if not np.isfinite(step) or step < 1 or not np.isclose(step, round(step), rtol=0, atol=1e-6):
        raise ValueError('bin_seconds must specify a positive integer sample width.')
    step = int(round(step))
    bins = (source.num_samples + step - 1) // step
    size = len(source.unit_ids) * bins * 8 * (1 + 2 * len(scalar_attributes(source)))
    size += (bins * 2 + 1) * 8
    return size, step, bins


def build_summaries(
    source,
    output_dir,
    bin_seconds=3600,
    max_bytes=536870912,
    workers=1,
    cancel=None,
    progress=None,
):
    """Stream exact full counts, exposure, and finite-value sums/counts.

    Exposure is one shared vector. Summing parent-unit counts on merge must
    divide by this vector once. Scalar means are sums/counts, never medians.
    """
    worker_count(workers)
    check_cancel(cancel)
    size, step, bins = summary_size(source, bin_seconds)
    if size + 1024 * (1 + 2 * len(scalar_attributes(source))) > max_bytes:
        raise ValueError(
            f'Summary budget exceeded: need at least {size} array bytes; '
            'increase bin_seconds or max_bytes.'
        )
    with ExitStack() as stack:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            'counts_file': '_phy_summary_counts.npy',
            'edges_file': '_phy_summary_edges.npy',
            'exposure_file': '_phy_summary_exposure.npy',
        }
        shape = (len(source.unit_ids), bins)
        counts = np.lib.format.open_memmap(
            output_dir / result['counts_file'], mode='w+', dtype=np.int64, shape=shape
        )
        stack.callback(close_array, counts)
        counts[:] = 0
        attributes = scalar_attributes(source)
        grids = {}
        for field in attributes:
            for kind, dtype in (('sum', np.float64), ('count', np.int64)):
                key = f'{field}_{kind}_file'
                result[key] = f'_phy_summary_{field}_{kind}.npy'
                array = np.lib.format.open_memmap(
                    output_dir / result[key], mode='w+', dtype=dtype, shape=shape
                )
                stack.callback(close_array, array)
                array[:] = 0
                grids[field, kind] = array
        edges = np.lib.format.open_memmap(
            output_dir / result['edges_file'], mode='w+', dtype=np.int64, shape=(bins + 1,)
        )
        stack.callback(close_array, edges)
        # Bounded chunks also avoid a recording-duration-sized temporary arange.
        for start in range(0, bins + 1, source.batch_size):
            stop = min(start + source.batch_size, bins + 1)
            edges[start:stop] = np.arange(start, stop, dtype=np.int64) * step
        edges[-1] = source.num_samples
        exposure = np.lib.format.open_memmap(
            output_dir / result['exposure_file'], mode='w+', dtype=np.float64, shape=(bins,)
        )
        stack.callback(close_array, exposure)
        exposure[:] = 0
        for start, end in source.valid_intervals:
            first, last = int(start // step), int((end - 1) // step)
            for offset in range(first, last + 1, source.batch_size):
                stop = min(offset + source.batch_size, last + 1)
                exposure[offset:stop] += (
                    np.minimum(edges[offset + 1 : stop + 1], end)
                    - np.maximum(edges[offset:stop], start)
                ) / source.sample_rate
        # Output grids and all simultaneous worker histograms share max_bytes.
        # Map only compact IDs; a sparse int32 catalog never allocates a huge LUT.
        available = max_bytes - size
        lookup = UnitRows(source.unit_ids, max_bytes=min(8 * 1024 * 1024, available // 8))
        if lookup.table is not None:
            available -= lookup.table.nbytes
        # Reserve one consumer result while the next worker is scheduled.
        per_worker = available // (workers + 1)
        fields = 1 + 2 * len(attributes)
        row_bytes = 128 + 64 * len(attributes)
        batch_rows = max(1, min(source.batch_size, per_worker // (2 * row_bytes)))
        tile_bins = min(
            max(1, batch_rows // max(1, len(source.unit_ids))),
            per_worker // max(1, 16 * fields * len(source.unit_ids)),
        )
        times = source.arrays['spike_times.npy'].reshape(-1)
        assignments = source.arrays['spike_clusters.npy'].reshape(-1)

        def tasks():
            for start in range(0, source.n_spikes, batch_rows):
                stop = min(start + batch_rows, source.n_spikes)
                first = start
                while first < stop:
                    begin = int(times[first] // step)
                    end = min(bins, begin + tile_bins, int(times[stop - 1] // step) + 1)
                    last = first + int(
                        np.searchsorted(times[first:stop], min(end * step, source.num_samples))
                    )
                    yield first, last, begin, end
                    first = last

        def histogram(task):
            check_cancel(cancel)
            first, last, begin, end = task
            rows = lookup(assignments[first:last])
            columns = times[first:last] // step - begin
            width = end - begin
            flat = rows.astype(np.int64) * width + columns
            cells = len(source.unit_ids) * width
            histograms = {None: np.bincount(flat, minlength=cells).reshape(-1, width)}
            for field, values in attributes.items():
                values = values[first:last]
                finite = np.isfinite(values)
                histograms[field, 'sum'] = np.bincount(
                    flat[finite], weights=values[finite].astype(np.float64), minlength=cells
                ).reshape(-1, width)
                histograms[field, 'count'] = np.bincount(flat[finite], minlength=cells).reshape(
                    -1, width
                )
            return last, begin, end, histograms

        if tile_bins:
            for stop, begin, end, histograms in bounded_map(histogram, tasks(), workers):
                check_cancel(cancel)
                counts[:, begin:end] += histograms.pop(None)
                for key, values in histograms.items():
                    grids[key][:, begin:end] += values
                if progress:
                    progress('building full summaries', stop, source.n_spikes)
        else:
            # Extremely tight budgets cannot fit even a unit-by-one-bin
            # histogram. Preserve the bounded scatter path in that case.
            for start in range(0, source.n_spikes, batch_rows):
                check_cancel(cancel)
                stop = min(start + batch_rows, source.n_spikes)
                rows = lookup(assignments[start:stop])
                columns = times[start:stop] // step
                np.add.at(counts, (rows, columns), 1)
                for field, values in attributes.items():
                    values = values[start:stop]
                    finite = np.isfinite(values)
                    np.add.at(grids[field, 'sum'], (rows[finite], columns[finite]), values[finite])
                    np.add.at(grids[field, 'count'], (rows[finite], columns[finite]), 1)
                if progress:
                    progress('building full summaries', stop, source.n_spikes)
        for array in (counts, edges, exposure, *grids.values()):
            array.flush()
        result['estimators'] = {
            'counts': 'exact',
            'exposure': 'acquisition_seconds',
            **dict.fromkeys(attributes, 'finite_sum_and_count'),
        }
        result['measurement_units'] = {field: f'source {field} units' for field in attributes}
        return result
