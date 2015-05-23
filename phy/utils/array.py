# -*- coding: utf-8 -*-

"""Utility functions for NumPy arrays."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from math import floor

import numpy as np

from ..ext.six import integer_types
from .logging import warn
from ._types import _as_tuple, _as_array


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
    if len(x) == 0:
        return np.array([], dtype=np.int64)
    bc = np.bincount(x)
    return np.nonzero(bc)[0]


def _ensure_unique(func):
    """Apply unique() to the output of a function."""
    def wrapped(*args, **kwargs):
        out = func(*args, **kwargs)
        return _unique(out)
    return wrapped


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
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m, dtype=np.int)
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


def _partial_shape(shape, trailing_index):
    """Return the shape of a partial array."""
    if shape is None:
        shape = ()
    if trailing_index is None:
        trailing_index = ()
    trailing_index = _as_tuple(trailing_index)
    # Length of the selection items for the partial array.
    len_item = len(shape) - len(trailing_index)
    # Array for the trailing dimensions.
    _arr = np.empty(shape=shape[len_item:])
    try:
        trailing_arr = _arr[trailing_index]
    except IndexError:
        raise ValueError("The partial shape index is invalid.")
    return shape[:len_item] + trailing_arr.shape


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
        # return item[0], item[-1]
        raise NotImplementedError()
    else:
        # Integer.
        return item, item + 1


def _len_index(item, max_len=0):
    """Return the expected length of the output of __getitem__(item)."""
    if isinstance(item, (list, np.ndarray)):
        return len(item)
    elif isinstance(item, slice):
        stop = item.stop or max_len
        start = item.start or 0
        step = item.step or 1
        start = np.clip(start, 0, stop)
        assert 0 <= start <= stop
        return 1 + ((stop - 1 - start) // step)
    else:
        return 1


def _fill_index(arr, item):
    if isinstance(item, tuple):
        item = (slice(None, None, None),) + item[1:]
        return arr[item]
    else:
        return arr


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
    if n_excerpts * excerpt_size > len(data):
        return data
    if n_excerpts == 1:
        return data
    return np.concatenate([data_chunk(data, chunk)
                           for chunk in excerpts(len(data),
                                                 n_excerpts=n_excerpts,
                                                 excerpt_size=excerpt_size)])


def regular_subset(spikes=None, n_spikes_max=None):
    """Prune the current selection to get at most n_spikes_max spikes."""
    assert spikes is not None
    # Nothing to do if the selection already satisfies n_spikes_max.
    if n_spikes_max is None or len(spikes) <= n_spikes_max:
        return spikes
    step = int(np.clip(1. / n_spikes_max * len(spikes),
                       1, len(spikes)))
    # Random shift.
    start = np.random.randint(low=0, high=step)
    my_spikes = spikes[start::step][:n_spikes_max]
    assert len(my_spikes) <= len(spikes)
    assert len(my_spikes) <= n_spikes_max
    return my_spikes


# -----------------------------------------------------------------------------
# Spike clusters utility functions
# -----------------------------------------------------------------------------

def _spikes_in_clusters(spike_clusters, clusters):
    """Return the ids of all spikes belonging to the specified clusters."""
    if len(spike_clusters) == 0 or len(clusters) == 0:
        return np.array([], dtype=np.int)
    return np.nonzero(np.in1d(spike_clusters, clusters))[0]


def _spikes_per_cluster(spike_ids, spike_clusters):
    """Return a dictionary {cluster: list_of_spikes}."""
    rel_spikes = np.argsort(spike_clusters)
    abs_spikes = spike_ids[rel_spikes]
    spike_clusters = spike_clusters[rel_spikes]

    diff = np.empty_like(spike_clusters)
    diff[0] = 1
    diff[1:] = np.diff(spike_clusters)

    idx = np.nonzero(diff > 0)[0]
    clusters = spike_clusters[idx]

    spikes_in_clusters = {clusters[i]: np.sort(abs_spikes[idx[i]:idx[i+1]])
                          for i in range(len(clusters) - 1)}
    spikes_in_clusters[clusters[-1]] = np.sort(abs_spikes[idx[-1]:])

    return spikes_in_clusters


def _flatten_spikes_per_cluster(spikes_per_cluster):
    """Convert a dictionary {cluster: list_of_spikes} to a
    spike_clusters array."""
    clusters = sorted(spikes_per_cluster)
    clusters_arr = np.concatenate([(cluster *
                                   np.ones(len(spikes_per_cluster[cluster])))
                                   for cluster in clusters]).astype(np.int64)
    spikes_arr = np.concatenate([spikes_per_cluster[cluster]
                                 for cluster in clusters])
    spike_clusters = np.vstack((spikes_arr, clusters_arr))
    ind = np.argsort(spike_clusters[0, :])
    return spike_clusters[1, ind]


def _concatenate_per_cluster_arrays(spikes_per_cluster, arrays):
    """Concatenate arrays from a {cluster: array} dictionary."""
    assert set(arrays) <= set(spikes_per_cluster)
    clusters = sorted(arrays)
    # Check the sizes of the spikes per cluster and the arrays.
    n_0 = [len(spikes_per_cluster[cluster]) for cluster in clusters]
    n_1 = [len(arrays[cluster]) for cluster in clusters]
    assert n_0 == n_1

    # Concatenate all spikes to find the right insertion order.
    spikes = np.concatenate([spikes_per_cluster[cluster]
                             for cluster in clusters])
    idx = np.argsort(spikes)
    # NOTE: concatenate all arrays along the first axis, because we assume
    # that the first axis represents the spikes.
    arrays = np.concatenate([_as_array(arrays[cluster])
                             for cluster in clusters])
    return arrays[idx, ...]


def _subset_spikes_per_cluster(spikes_per_cluster, arrays, spikes_sub,
                               allow_cut=False):
    """Cut spikes_per_cluster and arrays along a list of spikes."""
    # WARNING: spikes_sub should be sorted and without duplicates.
    spikes_sub = _as_array(spikes_sub)
    spikes_per_cluster_subset = {}
    arrays_subset = {}
    n = 0

    # Opt-in parameter to allow cutting the requested spikes with
    # the spikes per cluster dictionary.
    _all_spikes = np.hstack(spikes_per_cluster.values())
    if allow_cut:
        spikes_sub = np.intersect1d(spikes_sub,
                                    _all_spikes)
    assert np.all(np.in1d(spikes_sub, _all_spikes))

    for cluster in sorted(spikes_per_cluster):
        spikes_c = _as_array(spikes_per_cluster[cluster])
        array = _as_array(arrays[cluster])
        assert spikes_sub.dtype == spikes_c.dtype
        spikes_sc = np.intersect1d(spikes_sub, spikes_c)
        # assert spikes_sc.dtype == np.int64
        spikes_per_cluster_subset[cluster] = spikes_sc
        idx = _index_of(spikes_sc, spikes_c)
        arrays_subset[cluster] = array[idx, ...]
        assert len(spikes_sc) == len(arrays_subset[cluster])
        n += len(spikes_sc)
    assert n == len(spikes_sub)
    return spikes_per_cluster_subset, arrays_subset


def _flatten_per_cluster(arrs, spc=None):
    """Return an array from a dictionary `{cluster: data}`.

    There are three cases:

    * `data` is a scalar: return a `n_clusters` vector
    * `data` is an array: return a `(n_spikes, ...)` matrix
    * `data` is `(arr, spikes)`: return a `(n_spikes, ...)` matrix

    """
    assert isinstance(arrs, dict)
    clusters = sorted(arrs)
    if spc:
        assert isinstance(spc, dict)
        assert set(clusters) <= set(spc)

    # First case: scalar.
    if clusters and not isinstance(arrs[clusters[0]], (np.ndarray, tuple)):
        return np.array([arrs[cluster] for cluster in clusters])

    def _spikes_clusters(cluster, res):
        if isinstance(res, tuple) and len(res) == 2:
            arr, spk = res
            assert arr.shape[0] == len(spk)
            return arr, spk
        else:
            return res, spc[cluster]

    spc = {cluster: spk for cluster, (_, spk) in arrs.items()}
    arrays = {cluster: arr for cluster, (arr, _) in arrs.items()}
    return _concatenate_per_cluster_arrays(spc, arrays)


# -----------------------------------------------------------------------------
# PartialArray
# -----------------------------------------------------------------------------

class PartialArray(object):
    """Proxy to a view of an array, allowing selection along the first
    dimensions and fixing the trailing dimensions."""
    def __init__(self, arr, trailing_index=None):
        self._arr = arr
        self._trailing_index = _as_tuple(trailing_index)
        self.shape = _partial_shape(arr.shape, self._trailing_index)
        self.dtype = arr.dtype
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        if self._trailing_index is None:
            return self._arr[item]
        else:
            item = _as_tuple(item)
            k = len(item)
            n = len(self._arr.shape)
            t = len(self._trailing_index)
            if k < (n - t):
                item += (slice(None, None, None),) * (n - k - t)
            item += self._trailing_index
            if len(item) != n:
                raise ValueError("The array selection is invalid: "
                                 "{0}".format(str(item)))
            return self._arr[item]

    def __len__(self):
        return self.shape[0]


class ConcatenatedArrays(object):
    """This object represents a concatenation of several memory-mapped
    arrays."""
    def __init__(self, arrs):
        assert isinstance(arrs, list)
        self.arrs = arrs
        self.offsets = np.concatenate([[0], np.cumsum([arr.shape[0]
                                                       for arr in arrs])],
                                      axis=0)
        self.dtype = arrs[0].dtype if arrs else None
        self.shape = (self.offsets[-1],) + arrs[0].shape[1:]

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
        # Get the start and stop indices of the requested item.
        start, stop = _start_stop(item)
        # Return the concatenation of all arrays.
        if start is None and stop is None:
            return np.concatenate(self.arrs, axis=0)
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
            return _fill_index(self.arrs[rec_start][start_rel:stop_rel],
                               item)
        chunk_start = self.arrs[rec_start][start_rel:]
        chunk_stop = self.arrs[rec_stop][:stop_rel]
        # Concatenate all chunks.
        l = [chunk_start]
        if rec_stop - rec_start >= 2:
            warn("Loading a full virtual array: this might be slow "
                 "and something might be wrong.")
            l += [self.arrs[r][...] for r in range(rec_start + 1,
                                                   rec_stop)]
        l += [chunk_stop]
        # Apply the rest of the index.
        return _fill_index(np.concatenate(l, axis=0), item)


class VirtualMappedArray(object):
    """A virtual mapped array that yields null arrays to any selection."""
    def __init__(self, shape, dtype, fill=0):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(self.shape)
        self._fill = fill

    def __getitem__(self, item):
        if isinstance(item, integer_types):
            return self._fill * np.ones(self.shape[1:], dtype=self.dtype)
        else:
            assert not isinstance(item, tuple)
            n = _len_index(item, max_len=self.shape[0])
            return self._fill * np.ones((n,) + self.shape[1:],
                                        dtype=self.dtype)

    def __len__(self):
        return self.shape[0]


def _concatenate_virtual_arrays(arrs):
    """Return a virtual concatenate of several NumPy arrays."""
    n = len(arrs)
    if n == 1:
        return arrs[0]
    return ConcatenatedArrays(arrs)
