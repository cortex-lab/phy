"""Typed threshold boundaries and conservative float64 aggregation bounds."""

import numpy as np

from .curation import evaluate_predicate
from .range_intervals import RangeUnavailableError

EPS = np.finfo(np.float64).eps
SUM_RTOL = 1e-9
SUM_ATOL = 1e-6


def node_dtype(fields):
    return np.dtype(
        [
            (f'{field}_{kind}', '<i8' if kind == 'count' else '<f8')
            for field in fields
            for kind in ('sum', 'count', 'abs', 'error')
        ]
    )


def gamma(count):
    product = np.asarray(count, dtype=np.float64) * EPS
    return np.divide(
        product, 1 - product, where=product < 1, out=np.full(np.shape(product), np.inf)
    )


def row_stats(records, fields, dtype, valid=None):
    """One or many blocks; nonfinite exclusions follow the original raw dtype."""
    if records.ndim == 1:
        records = records[None, :]
    output = np.zeros(records.shape[0], dtype=dtype)
    for field in fields:
        raw = records[field]
        finite = np.isfinite(raw)
        if valid is not None:
            finite &= valid
        values = np.where(finite, raw, 0).astype(np.float64)
        count = finite.sum(axis=1, dtype=np.int64)
        with np.errstate(over='ignore', invalid='ignore'):
            absolute = np.nextafter(np.abs(values).sum(axis=1) / (1 - gamma(count)), np.inf)
            output[field + '_sum'] = values.sum(axis=1, dtype=np.float64)
            output[field + '_count'] = count
            output[field + '_abs'] = absolute
            output[field + '_error'] = np.nextafter(gamma(count) * absolute, np.inf)
    return output


def add_stats(left, right, fields):
    """Combine disjoint aggregates without subtracting large floating prefixes."""
    result = np.zeros(np.broadcast_shapes(left.shape, right.shape), dtype=left.dtype)
    for field in fields:
        with np.errstate(over='ignore', invalid='ignore'):
            absolute = np.nextafter(left[field + '_abs'] + right[field + '_abs'], np.inf)
            result[field + '_sum'] = left[field + '_sum'] + right[field + '_sum']
            result[field + '_count'] = left[field + '_count'] + right[field + '_count']
            result[field + '_abs'] = absolute
            result[field + '_error'] = np.nextafter(
                left[field + '_error'] + right[field + '_error'] + EPS * absolute, np.inf
            )
    return result


def validate_sums(summary, absolute, error, operations, fields):
    """Fall back if reordering cannot meet the existing trend-sum tolerance."""
    for field in fields:
        counts = summary['trend_counts'][field]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            correction = gamma(operations[field])
            upper_abs = np.nextafter(absolute[field] / (1 - correction), np.inf)
            tree_error = error[field] / (1 - correction) + correction * upper_abs
            bound = gamma(counts) * upper_abs + tree_error
            allowed = SUM_ATOL + SUM_RTOL * np.maximum(
                0, np.abs(summary['trend_sums'][field]) - bound
            )
        if np.any(~np.isfinite(bound)) or np.any(bound > allowed):
            raise RangeUnavailableError('Trend arithmetic needs the verified row fallback.')


def typed_rank(values, predicate):
    """Reference scalar search using the same array ufunc as semantic replay.

    Deliberately avoid searchsorted's different scalar-promotion rules.
    """
    lo, hi = 0, len(values)
    while lo < hi:
        middle = (lo + hi) // 2
        if evaluate_predicate(values[middle : middle + 1], predicate)[0]:
            lo = middle + 1
        else:
            hi = middle
    return lo
