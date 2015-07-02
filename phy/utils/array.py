# -*- coding: utf-8 -*-

"""Utility functions for NumPy arrays."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
from math import floor
from operator import mul
from functools import reduce
import math

import numpy as np

from ..ext.six import integer_types, string_types
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
    if x is None or len(x) == 0:
        return np.array([], dtype=np.int64)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _as_array(x)
    x = x[x >= 0]
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
    tmp = np.zeros(m + 1, dtype=np.int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


def index_of_small(arr, lookup):
    """Faster on small arrays with large values."""
    return np.searchsorted(lookup, arr)


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
        return np.min(item), np.max(item)
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
# I/O functions
# -----------------------------------------------------------------------------

def _prod(l):
    return reduce(mul, l, 1)


class LazyMemmap(object):
    """A memmapped array that only opens the file handle when required."""
    def __init__(self, path, dtype=None, shape=None, mode='r'):
        assert isinstance(path, string_types)
        assert dtype
        self._path = path
        self._f = None
        self.mode = mode
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape) if shape else None

    def _open_file_if_needed(self):
        if self._f is None:
            self._f = np.memmap(self._path,
                                dtype=self.dtype,
                                shape=self.shape,
                                mode=self.mode,
                                )
            self.shape = self._f.shape
            self.ndim = self._f.ndim

    def __getitem__(self, item):
        self._open_file_if_needed()
        return self._f[item]

    def __len__(self):
        self._open_file_if_needed()
        return len(self._f)

    def __del__(self):
        if self._f is not None:
            del self._f


def _memmap(f_or_path, dtype=None, shape=None, lazy=True):
    if not lazy:
        return np.memmap(f_or_path, dtype=dtype, shape=shape, mode='r')
    else:
        return LazyMemmap(f_or_path, dtype=dtype, shape=shape, mode='r')


def _file_size(f_or_path):
    if isinstance(f_or_path, string_types):
        with open(f_or_path, 'rb') as f:
            return _file_size(f)
    else:
        return os.fstat(f_or_path.fileno()).st_size


def _load_ndarray(f_or_path, dtype=None, shape=None, memmap=False, lazy=False):
    if dtype is None:
        return f_or_path
    else:
        if not memmap:
            arr = np.fromfile(f_or_path, dtype=dtype)
            if shape is not None:
                arr = arr.reshape(shape)
        else:
            # memmap doesn't accept -1 in shapes, but we can compute
            # the missing dimension from the file size.
            if shape and shape[0] == -1:
                n_bytes = _file_size(f_or_path)
                n_items = n_bytes // np.dtype(dtype).itemsize
                n_rows = n_items // _prod(shape[1:])
                shape = (n_rows,) + shape[1:]
                assert _prod(shape) == n_items
            arr = _memmap(f_or_path, dtype=dtype, shape=shape, lazy=lazy)
        return arr


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


def regular_subset(spikes=None, n_spikes_max=None):
    """Prune the current selection to get at most n_spikes_max spikes."""
    assert spikes is not None
    # Nothing to do if the selection already satisfies n_spikes_max.
    if n_spikes_max is None or len(spikes) <= n_spikes_max:
        return spikes
    step = math.ceil(np.clip(1. / n_spikes_max * len(spikes),
                             1, len(spikes)))
    step = int(step)
    # Random shift.
    # start = np.random.randint(low=0, high=step)
    # Note: randomly-changing selections are confusing...
    start = 0
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
    # spikes_per_cluster case.
    if isinstance(spike_clusters, dict):
        return np.sort(np.concatenate([spike_clusters[cluster]
                                       for cluster in clusters]))
    return np.nonzero(np.in1d(spike_clusters, clusters))[0]


def _spikes_per_cluster(spike_ids, spike_clusters):
    """Return a dictionary {cluster: list_of_spikes}."""
    if not len(spike_ids):
        return {}
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
    if not len(clusters):
        return np.array([])

    spikes = np.concatenate([spikes_per_cluster[cluster]
                             for cluster in clusters])
    idx = np.argsort(spikes)
    # NOTE: concatenate all arrays along the first axis, because we assume
    # that the first axis represents the spikes.
    arrays = np.concatenate([_as_array(arrays[cluster])
                             for cluster in clusters])
    return arrays[idx, ...]


def _subset_spc(spc, clusters):
    return {c: s for c, s in spc.items()
            if c in clusters}


class PerClusterData(object):
    """Store data associated to every spike.

    This class provides several data structures, with per-spike data and
    per-cluster data. It also defines a `subset()` method that allows to
    make a subset of the data using either spikes or clusters.

    """
    def __init__(self,
                 spike_ids=None, array=None, spike_clusters=None,
                 spc=None, arrays=None):
        if (array is not None and spike_ids is not None):
            # From array to per-cluster arrays.
            self._spike_ids = _as_array(spike_ids)
            self._array = _as_array(array)
            self._spike_clusters = _as_array(spike_clusters)
            self._check_array()
            self._split()
            self._check_dict()
        elif (arrays is not None and spc is not None):
            # From per-cluster arrays to array.
            self._spc = spc
            self._arrays = arrays
            self._check_dict()
            self._concatenate()
            self._check_array()
        else:
            raise ValueError()

    @property
    def spike_ids(self):
        """Sorted array of all spike ids."""
        return self._spike_ids

    @property
    def spike_clusters(self):
        """Array with the cluster id of every spike."""
        return self._spike_clusters

    @property
    def array(self):
        """Data array.

        The first dimension of the array corresponds to the spikes in the
        cluster.

        """
        return self._array

    @property
    def arrays(self):
        """Dictionary of arrays `{cluster: array}`.

        The first dimension of the arrays correspond to the spikes in the
        cluster.

        """
        return self._arrays

    @property
    def spc(self):
        """Spikes per cluster dictionary."""
        return self._spc

    @property
    def cluster_ids(self):
        """Sorted list of clusters."""
        return self._cluster_ids

    @property
    def n_clusters(self):
        return len(self._cluster_ids)

    def _check_dict(self):
        assert set(self._arrays) == set(self._spc)
        clusters = sorted(self._arrays)
        n_0 = [len(self._spc[cluster]) for cluster in clusters]
        n_1 = [len(self._arrays[cluster]) for cluster in clusters]
        assert n_0 == n_1

    def _check_array(self):
        assert len(self._array) == len(self._spike_ids)
        assert len(self._spike_clusters) == len(self._spike_ids)

    def _concatenate(self):
        self._cluster_ids = sorted(self._spc)
        n = len(self.cluster_ids)
        if n == 0:
            self._array = np.array([])
            self._spike_clusters = np.array([], dtype=np.int32)
            self._spike_ids = np.array([], dtype=np.int64)
        elif n == 1:
            c = self.cluster_ids[0]
            self._array = _as_array(self._arrays[c])
            self._spike_ids = self._spc[c]
            self._spike_clusters = c * np.ones(len(self._spike_ids),
                                               dtype=np.int32)
        else:
            # Concatenate all spikes to find the right insertion order.
            spikes = np.concatenate([self._spc[cluster]
                                     for cluster in self.cluster_ids])
            idx = np.argsort(spikes)
            self._spike_ids = np.sort(spikes)
            # NOTE: concatenate all arrays along the first axis, because we
            # assume that the first axis represents the spikes.
            # TODO OPTIM: use ConcatenatedArray and implement custom indices.
            # array = ConcatenatedArrays([_as_array(self._arrays[cluster])
            #                             for cluster in self.cluster_ids])
            array = np.concatenate([_as_array(self._arrays[cluster])
                                    for cluster in self.cluster_ids])
            self._array = array[idx]
            self._spike_clusters = _flatten_spikes_per_cluster(self._spc)

    def _split(self):
        self._spc = _spikes_per_cluster(self._spike_ids,
                                        self._spike_clusters)
        self._cluster_ids = sorted(self._spc)
        n = len(self.cluster_ids)
        # Optimization for single cluster.
        if n == 0:
            self._arrays = {}
        elif n == 1:
            c = self._cluster_ids[0]
            self._arrays = {c: self._array}
        else:
            self._arrays = {}
            for cluster in sorted(self._cluster_ids):
                spk = _as_array(self._spc[cluster])
                spk_rel = _index_of(spk, self._spike_ids)
                self._arrays[cluster] = self._array[spk_rel]

    def subset(self, spike_ids=None, clusters=None, spc=None):
        """Return a new PerClusterData instance with a subset of the data.

        There are three ways to specify the subset:

        * With a list of spikes
        * With a list of clusters
        * With a dictionary of `{cluster: some_spikes}`

        """
        if spike_ids is not None:
            if np.array_equal(spike_ids, self._spike_ids):
                return self
            assert np.all(np.in1d(spike_ids, self._spike_ids))
            spike_ids_s_rel = _index_of(spike_ids, self._spike_ids)
            array_s = self._array[spike_ids_s_rel]
            spike_clusters_s = self._spike_clusters[spike_ids_s_rel]
            return PerClusterData(spike_ids=spike_ids,
                                  array=array_s,
                                  spike_clusters=spike_clusters_s,
                                  )
        elif clusters is not None:
            assert set(clusters) <= set(self._cluster_ids)
            spc_s = {clu: self._spc[clu] for clu in clusters}
            arrays_s = {clu: self._arrays[clu] for clu in clusters}
            return PerClusterData(spc=spc_s, arrays=arrays_s)
        elif spc is not None:
            clusters = sorted(spc)
            assert set(clusters) <= set(self._cluster_ids)
            arrays_s = {}
            for cluster in clusters:
                spk_rel = _index_of(_as_array(spc[cluster]),
                                    _as_array(self._spc[cluster]))
                arrays_s[cluster] = _as_array(self._arrays[cluster])[spk_rel]
            return PerClusterData(spc=spc, arrays=arrays_s)


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

    def __len__(self):
        return self.shape[0]


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
    if n == 0:
        return None
    elif n == 1:
        return arrs[0]
    return ConcatenatedArrays(arrs)
