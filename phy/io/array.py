# -*- coding: utf-8 -*-

"""Utility functions for NumPy arrays."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import math
from math import floor, exp
import os.path as op

import numpy as np

from phy.utils._types import _as_array, _is_array_like

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _range_from_slice(myslice, start=None, stop=None, step=None, length=None):
    """Convert a slice to an array of integers."""
    assert isinstance(myslice, slice)
    # Find 'step'.
    step = step if step is not None else myslice.step
    if step is None:
        step = 1
    # Find 'start'.
    start = start if start is not None else myslice.start
    if start is None:
        start = 0
    # Find 'stop' as a function of length if 'stop' is unspecified.
    stop = stop if stop is not None else myslice.stop
    if length is not None:
        stop_inferred = floor(start + step * length)
        if stop is not None and stop < stop_inferred:
            raise ValueError("'stop' ({stop}) and ".format(stop=stop) +
                             "'length' ({length}) ".format(length=length) +
                             "are not compatible.")
        stop = stop_inferred
    if stop is None and length is None:
        raise ValueError("'stop' and 'length' cannot be both unspecified.")
    myrange = np.arange(start, stop, step)
    # Check the length if it was specified.
    if length is not None:
        assert len(myrange) == length
    return myrange


def _unique(x):
    """Faster version of np.unique().

    This version is restricted to 1D arrays of non-negative integers.

    It is only faster if len(x) >> len(unique(x)).

    """
    if x is None or len(x) == 0:
        return np.array([], dtype=np.int64)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _as_array(x)
    x = x[x >= 0]
    bc = np.bincount(x)
    return np.nonzero(bc)[0]


def _normalize(arr, keep_ratio=False):
    """Normalize an array into [0, 1]."""
    (x_min, y_min), (x_max, y_max) = arr.min(axis=0), arr.max(axis=0)

    if keep_ratio:
        a = 1. / max(x_max - x_min, y_max - y_min)
        ax = ay = a
        bx = .5 - .5 * a * (x_max + x_min)
        by = .5 - .5 * a * (y_max + y_min)
    else:
        ax = 1. / (x_max - x_min)
        ay = 1. / (y_max - y_min)
        bx = -x_min / (x_max - x_min)
        by = -y_min / (y_max - y_min)

    arr_n = arr.copy()
    arr_n[:, 0] *= ax
    arr_n[:, 0] += bx
    arr_n[:, 1] *= ay
    arr_n[:, 1] += by

    return arr_n


def _index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.

    Implicitely assume that:

    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.

    This is not checked for performance reasons.

    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    # TODO: assertions to disable in production for performance reasons.
    # TODO: np.searchsorted(lookup, arr) is faster on small arrays with large
    # values
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=np.int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


def _pad(arr, n, dir='right'):
    """Pad an array with zeros along the first axis.

    Parameters
    ----------

    n : int
        Size of the returned array in the first axis.
    dir : str
        Direction of the padding. Must be one 'left' or 'right'.

    """
    assert dir in ('left', 'right')
    if n < 0:
        raise ValueError("'n' must be positive: {0}.".format(n))
    elif n == 0:
        return np.zeros((0,) + arr.shape[1:], dtype=arr.dtype)
    n_arr = arr.shape[0]
    shape = (n,) + arr.shape[1:]
    if n_arr == n:
        assert arr.shape == shape
        return arr
    elif n_arr < n:
        out = np.zeros(shape, dtype=arr.dtype)
        if dir == 'left':
            out[-n_arr:, ...] = arr
        elif dir == 'right':
            out[:n_arr, ...] = arr
        assert out.shape == shape
        return out
    else:
        if dir == 'left':
            out = arr[-n:, ...]
        elif dir == 'right':
            out = arr[:n, ...]
        assert out.shape == shape
        return out


def _get_padded(data, start, end):
    """Return `data[start:end]` filling in with zeros outside array bounds

    Assumes that either `start<0` or `end>len(data)` but not both.

    """
    if start < 0 and end > data.shape[0]:
        raise RuntimeError()
    if start < 0:
        start_zeros = np.zeros((-start, data.shape[1]),
                               dtype=data.dtype)
        return np.vstack((start_zeros, data[:end]))
    elif end > data.shape[0]:
        end_zeros = np.zeros((end - data.shape[0], data.shape[1]),
                             dtype=data.dtype)
        return np.vstack((data[start:], end_zeros))
    else:
        return data[start:end]


def _in_polygon(points, polygon):
    """Return the points that are inside a polygon."""
    from matplotlib.path import Path
    points = _as_array(points)
    polygon = _as_array(polygon)
    assert points.ndim == 2
    assert polygon.ndim == 2
    path = Path(polygon, closed=True)
    return path.contains_points(points)


# -----------------------------------------------------------------------------
# I/O functions
# -----------------------------------------------------------------------------

def read_array(path, mmap_mode=None):
    """Read a .npy array."""
    file_ext = op.splitext(path)[1]
    if file_ext == '.npy':
        return np.load(path, mmap_mode=mmap_mode)
    raise NotImplementedError("The file extension `{}` ".format(file_ext) +
                              "is not currently supported.")


def write_array(path, arr):
    """Write an array to a .npy file."""
    file_ext = op.splitext(path)[1]
    if file_ext == '.npy':
        try:
            # Save a dask array into a .npy file chunk-by-chunk.
            from dask.array import Array, store
            if isinstance(arr, Array):
                f = np.memmap(path, mode='w+',
                              dtype=arr.dtype, shape=arr.shape)
                store(arr, f)
                del f
        except ImportError:  # pragma: no cover
            # We'll save the dask array normally: it works but it is less
            # efficient since we need to load everything in memory.
            pass
        return np.save(path, arr)
    raise NotImplementedError("The file extension `{}` ".format(file_ext) +
                              "is not currently supported.")


# -----------------------------------------------------------------------------
# Chunking functions
# -----------------------------------------------------------------------------

def _excerpt_step(n_samples, n_excerpts=None, excerpt_size=None):
    """Compute the step of an excerpt set as a function of the number
    of excerpts or their sizes."""
    assert n_excerpts >= 2
    step = max((n_samples - excerpt_size) // (n_excerpts - 1),
               excerpt_size)
    return step


def chunk_bounds(n_samples, chunk_size, overlap=0):
    """Return chunk bounds.

    Chunks have the form:

        [ overlap/2 | chunk_size-overlap | overlap/2 ]
        s_start   keep_start           keep_end     s_end

    Except for the first and last chunks which do not have a left/right
    overlap.

    This generator yields (s_start, s_end, keep_start, keep_end).

    """
    s_start = 0
    s_end = chunk_size
    keep_start = s_start
    keep_end = s_end - overlap // 2
    yield s_start, s_end, keep_start, keep_end

    while s_end - overlap + chunk_size < n_samples:
        s_start = s_end - overlap
        s_end = s_start + chunk_size
        keep_start = keep_end
        keep_end = s_end - overlap // 2
        if s_start < s_end:
            yield s_start, s_end, keep_start, keep_end

    s_start = s_end - overlap
    s_end = n_samples
    keep_start = keep_end
    keep_end = s_end
    if s_start < s_end:
        yield s_start, s_end, keep_start, keep_end


def excerpts(n_samples, n_excerpts=None, excerpt_size=None):
    """Yield (start, end) where start is included and end is excluded."""
    assert n_excerpts >= 2
    step = _excerpt_step(n_samples,
                         n_excerpts=n_excerpts,
                         excerpt_size=excerpt_size)
    for i in range(n_excerpts):
        start = i * step
        if start >= n_samples:
            break
        end = min(start + excerpt_size, n_samples)
        yield start, end


def data_chunk(data, chunk, with_overlap=False):
    """Get a data chunk."""
    assert isinstance(chunk, tuple)
    if len(chunk) == 2:
        i, j = chunk
    elif len(chunk) == 4:
        if with_overlap:
            i, j = chunk[:2]
        else:
            i, j = chunk[2:]
    else:
        raise ValueError("'chunk' should have 2 or 4 elements, "
                         "not {0:d}".format(len(chunk)))
    return data[i:j, ...]


def get_excerpts(data, n_excerpts=None, excerpt_size=None):
    assert n_excerpts is not None
    assert excerpt_size is not None
    if len(data) < n_excerpts * excerpt_size:
        return data
    elif n_excerpts == 0:
        return data[:0]
    elif n_excerpts == 1:
        return data[:excerpt_size]
    out = np.concatenate([data_chunk(data, chunk)
                          for chunk in excerpts(len(data),
                                                n_excerpts=n_excerpts,
                                                excerpt_size=excerpt_size)])
    assert len(out) <= n_excerpts * excerpt_size
    return out


# -----------------------------------------------------------------------------
# Spike clusters utility functions
# -----------------------------------------------------------------------------

def _spikes_in_clusters(spike_clusters, clusters):
    """Return the ids of all spikes belonging to the specified clusters."""
    if len(spike_clusters) == 0 or len(clusters) == 0:
        return np.array([], dtype=np.int)
    return np.nonzero(np.in1d(spike_clusters, clusters))[0]


def _spikes_per_cluster(spike_clusters, spike_ids=None):
    """Return a dictionary {cluster: list_of_spikes}."""
    if spike_clusters is None or not len(spike_clusters):
        return {}
    if spike_ids is None:
        spike_ids = np.arange(len(spike_clusters)).astype(np.int64)
    rel_spikes = np.argsort(spike_clusters)
    abs_spikes = spike_ids[rel_spikes]
    spike_clusters = spike_clusters[rel_spikes]

    diff = np.empty_like(spike_clusters)
    diff[0] = 1
    diff[1:] = np.diff(spike_clusters)

    idx = np.nonzero(diff > 0)[0]
    clusters = spike_clusters[idx]

    spikes_in_clusters = {clusters[i]: np.sort(abs_spikes[idx[i]:idx[i + 1]])
                          for i in range(len(clusters) - 1)}
    spikes_in_clusters[clusters[-1]] = np.sort(abs_spikes[idx[-1]:])

    return spikes_in_clusters


def _flatten_per_cluster(per_cluster):
    """Convert a dictionary {cluster: spikes} to a spikes array."""
    return np.sort(np.concatenate(list(per_cluster.values()))).astype(np.int64)


def grouped_mean(arr, spike_clusters):
    """Compute the mean of a spike-dependent quantity for every cluster.

    The two arguments should be 1D array with `n_spikes` elements.

    The output is a 1D array with `n_clusters` elements. The clusters are
    sorted in increasing order.

    """
    arr = np.asarray(arr)
    spike_clusters = np.asarray(spike_clusters)
    assert arr.ndim == 1
    assert arr.shape[0] == len(spike_clusters)
    cluster_ids = _unique(spike_clusters)
    spike_clusters_rel = _index_of(spike_clusters, cluster_ids)
    spike_counts = np.bincount(spike_clusters_rel)
    assert len(spike_counts) == len(cluster_ids)
    t = np.zeros(len(cluster_ids))
    # Compute the sum with possible repetitions.
    np.add.at(t, spike_clusters_rel, arr)
    return t / spike_counts


def regular_subset(spikes, n_spikes_max=None, offset=0):
    """Prune the current selection to get at most n_spikes_max spikes."""
    assert spikes is not None
    # Nothing to do if the selection already satisfies n_spikes_max.
    if n_spikes_max is None or len(spikes) <= n_spikes_max:  # pragma: no cover
        return spikes
    step = math.ceil(np.clip(1. / n_spikes_max * len(spikes),
                             1, len(spikes)))
    step = int(step)
    # Note: randomly-changing selections are confusing...
    my_spikes = spikes[offset::step][:n_spikes_max]
    assert len(my_spikes) <= len(spikes)
    assert len(my_spikes) <= n_spikes_max
    return my_spikes


def select_spikes(cluster_ids=None,
                  max_n_spikes_per_cluster=None,
                  spikes_per_cluster=None):
    """Return a selection of spikes belonging to the specified clusters."""
    assert _is_array_like(cluster_ids)
    if not len(cluster_ids):
        return np.array([], dtype=np.int64)
    if max_n_spikes_per_cluster in (None, 0):
        selection = {c: spikes_per_cluster[c] for c in cluster_ids}
    else:
        assert max_n_spikes_per_cluster > 0
        selection = {}
        n_clusters = len(cluster_ids)
        for cluster in cluster_ids:
            # Decrease the number of spikes per cluster when there
            # are more clusters.
            n = int(max_n_spikes_per_cluster * exp(-.1 * (n_clusters - 1)))
            n = max(1, n)
            spikes = spikes_per_cluster[cluster]
            selection[cluster] = regular_subset(spikes, n_spikes_max=n)
    return _flatten_per_cluster(selection)
