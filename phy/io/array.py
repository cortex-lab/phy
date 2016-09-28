# -*- coding: utf-8 -*-

"""Utility functions for NumPy arrays."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict
from functools import wraps
import logging
import math
from math import floor, exp
from operator import itemgetter
import os.path as op

import numpy as np

from phy.utils import _as_scalar, _as_scalars
from phy.utils._types import _as_array, _is_array_like

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _range_from_slice(myslice, start=None, stop=None, step=None, length=None):
    """Convert a slice to an array of integers."""
    assert isinstance(myslice, slice)
    # Find 'step'.
    step = myslice.step if myslice.step is not None else step
    if step is None:
        step = 1
    # Find 'start'.
    start = myslice.start if myslice.start is not None else start
    if start is None:
        start = 0
    # Find 'stop' as a function of length if 'stop' is unspecified.
    stop = myslice.stop if myslice.stop is not None else stop
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
    if len(polygon):
        polygon = np.vstack((polygon, polygon[0]))
    path = Path(polygon, closed=True)
    return path.contains_points(points)


def _get_data_lim(arr, n_spikes=None):
    n = arr.shape[0]
    k = max(1, n // n_spikes) if n_spikes else 1
    arr = np.abs(arr[::k])
    n = arr.shape[0]
    arr = arr.reshape((n, -1))
    return arr.max() or 1.


def get_closest_clusters(cluster_id, cluster_ids, sim_func, max_n=None):
    """Return a list of pairs `(cluster, similarity)` sorted by decreasing
    similarity to a given cluster."""
    l = [(_as_scalar(candidate), _as_scalar(sim_func(cluster_id, candidate)))
         for candidate in _as_scalars(cluster_ids)]
    l = sorted(l, key=itemgetter(1), reverse=True)
    return l[:max_n]


def concat_per_cluster(f):
    """Take a function accepting a single cluster, and return a function
    accepting multiple clusters."""
    @wraps(f)
    def wrapped(cluster_ids, **kwargs):
        # Single cluster.
        if not hasattr(cluster_ids, '__len__'):
            return f(cluster_ids, **kwargs)
        # Return the list of cluster-dependent objects.
        out = [f(c, **kwargs) for c in cluster_ids]
        # Flatten list of lists.
        if all(isinstance(_, list) for _ in out):
            out = _flatten(out)  # pragma: no cover
        return out
    return wrapped


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
        return np.save(path, arr)
    raise NotImplementedError("The file extension `{}` ".format(file_ext) +
                              "is not currently supported.")


# -----------------------------------------------------------------------------
# Virtual concatenation
# -----------------------------------------------------------------------------

def _start_stop(item):
    """Find the start and stop indices of a __getitem__ item.

    This is used only by ConcatenatedArrays.

    Only two cases are supported currently:

    * Single integer.
    * Contiguous slice in the first dimension only.

    """
    if isinstance(item, tuple):
        item = item[0]
    if isinstance(item, slice):
        # Slice.
        if item.step not in (None, 1):
            return NotImplementedError()
        return item.start, item.stop
    elif isinstance(item, (list, np.ndarray)):
        # List or array of indices.
        return np.min(item), np.max(item)
    else:
        # Integer.
        return item, item + 1


def _fill_index(arr, item):
    if isinstance(item, tuple):
        item = (slice(None, None, None),) + item[1:]
        return arr[item]
    else:
        return arr


class ConcatenatedArrays(object):
    """This object represents a concatenation of several memory-mapped
    arrays."""
    def __init__(self, arrs, cols=None, scaling=None):
        assert isinstance(arrs, list)
        self.arrs = arrs
        # Reordering of the columns.
        self.cols = cols
        self.offsets = np.concatenate([[0], np.cumsum([arr.shape[0]
                                                       for arr in arrs])],
                                      axis=0)
        self.dtype = arrs[0].dtype if arrs else None

        if scaling is None:
            return

        # Multiply the output of a function by some scaling.
        def _wrap(f):
            def wrapped(*args):
                return f(*args) * self.scaling
            return wrapped

        self.__getitem__ = _wrap(self.__getitem__)

    @property
    def shape(self):
        if self.arrs[0].ndim == 1:
            return (self.offsets[-1],)
        ncols = (len(self.cols) if self.cols is not None
                 else self.arrs[0].shape[1])
        return (self.offsets[-1], ncols)

    def _get_recording(self, index):
        """Return the recording that contains a given index."""
        assert index >= 0
        recs = np.nonzero((index - self.offsets[:-1]) >= 0)[0]
        if len(recs) == 0:
            # If the index is greater than the total size,
            # return the last recording.
            return len(self.arrs) - 1
        # Return the last recording such that the index is greater than
        # its offset.
        return recs[-1]

    def __getitem__(self, item):
        cols = self.cols if self.cols is not None else slice(None, None, None)
        # Get the start and stop indices of the requested item.
        start, stop = _start_stop(item)
        # Return the concatenation of all arrays.
        if start is None and stop is None:
            return np.concatenate(self.arrs, axis=0)[..., cols]
        if start is None:
            start = 0
        if stop is None:
            stop = self.offsets[-1]
        if stop < 0:
            stop = self.offsets[-1] + stop
        # Get the recording indices of the first and last item.
        rec_start = self._get_recording(start)
        rec_stop = self._get_recording(stop)
        assert 0 <= rec_start <= rec_stop < len(self.arrs)
        # Find the start and stop relative to the arrays.
        start_rel = start - self.offsets[rec_start]
        stop_rel = stop - self.offsets[rec_stop]
        # Single array case.
        if rec_start == rec_stop:
            # Apply the rest of the index.
            out = _fill_index(self.arrs[rec_start][start_rel:stop_rel], item)
            out = out[..., cols]
            return out
        chunk_start = self.arrs[rec_start][start_rel:]
        chunk_stop = self.arrs[rec_stop][:stop_rel]
        # Concatenate all chunks.
        l = [chunk_start]
        if rec_stop - rec_start >= 2:
            logger.warn("Loading a full virtual array: this might be slow "
                        "and something might be wrong.")
            l += [self.arrs[r][...] for r in range(rec_start + 1,
                                                   rec_stop)]
        l += [chunk_stop]
        # Apply the rest of the index.
        return _fill_index(np.concatenate(l, axis=0), item)[..., cols]

    def __len__(self):
        return self.shape[0]


def _concatenate_virtual_arrays(arrs, cols=None, scaling=None):
    """Return a virtual concatenate of several NumPy arrays."""
    n = len(arrs)
    if n == 0:
        return None
    return ConcatenatedArrays(arrs, cols, scaling=scaling)


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
                  spikes_per_cluster=None,
                  batch_size=None,
                  ):
    """Return a selection of spikes belonging to the specified clusters."""
    assert _is_array_like(cluster_ids)
    if not len(cluster_ids):
        return np.array([], dtype=np.int64)
    if max_n_spikes_per_cluster in (None, 0):
        selection = {c: spikes_per_cluster(c) for c in cluster_ids}
    else:
        assert max_n_spikes_per_cluster > 0
        selection = {}
        n_clusters = len(cluster_ids)
        for cluster in cluster_ids:
            # Decrease the number of spikes per cluster when there
            # are more clusters.
            n = int(max_n_spikes_per_cluster * exp(-.1 * (n_clusters - 1)))
            n = max(1, n)
            spikes = spikes_per_cluster(cluster)
            # Regular subselection.
            if batch_size is None or len(spikes) <= max(batch_size, n):
                spikes = regular_subset(spikes, n_spikes_max=n)
            else:
                # Batch selections of spikes.
                spikes = get_excerpts(spikes, n // batch_size, batch_size)
            selection[cluster] = spikes
    return _flatten_per_cluster(selection)


class Selector(object):
    """This object is passed with the `select` event when clusters are
    selected. It allows to make selections of spikes."""
    def __init__(self, spikes_per_cluster):
        # NOTE: spikes_per_cluster is a function.
        self.spikes_per_cluster = spikes_per_cluster

    def select_spikes(self, cluster_ids=None,
                      max_n_spikes_per_cluster=None,
                      batch_size=None,
                      ):
        if cluster_ids is None or not len(cluster_ids):
            return None
        ns = max_n_spikes_per_cluster
        assert len(cluster_ids) >= 1
        # Select a subset of the spikes.
        return select_spikes(cluster_ids,
                             spikes_per_cluster=self.spikes_per_cluster,
                             max_n_spikes_per_cluster=ns,
                             batch_size=batch_size,
                             )


# -----------------------------------------------------------------------------
# Accumulator
# -----------------------------------------------------------------------------

def _flatten(l):
    return [item for sublist in l for item in sublist]


class Accumulator(object):
    """Accumulate arrays for concatenation."""
    def __init__(self):
        self._data = defaultdict(list)

    def add(self, name, val):
        """Add an array."""
        self._data[name].append(val)

    def get(self, name):
        """Return the list of arrays for a given name."""
        return _flatten(self._data[name])

    @property
    def names(self):
        """List of names."""
        return set(self._data)

    def __getitem__(self, name):
        """Concatenate all arrays with a given name."""
        l = self._data[name]
        # Process scalars: only return the first one and don't concatenate.
        if len(l) and not hasattr(l[0], '__len__'):
            return l[0]
        return np.concatenate(l, axis=0)


def _accumulate(data_list, no_concat=()):
    """Concatenate a list of dicts `(name, array)`.

    You can specify some names which arrays should not be concatenated.
    This is necessary with lists of plots with different sizes.

    """
    acc = Accumulator()
    for data in data_list:
        for name, val in data.items():
            acc.add(name, val)
    out = {name: acc[name] for name in acc.names if name not in no_concat}

    # Some variables should not be concatenated but should be kept as lists.
    # This is when there can be several arrays of variable length (NumPy
    # doesn't support ragged arrays).
    out.update({name: acc.get(name) for name in no_concat})
    return out
