# -*- coding: utf-8 -*-

"""Tests of array utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from pytest import raises, mark

from .._types import _as_array
from ..array import (_unique,
                     _normalize,
                     _index_of,
                     _in_polygon,
                     _spikes_in_clusters,
                     _spikes_per_cluster,
                     _flatten_spikes_per_cluster,
                     _concatenate_per_cluster_arrays,
                     chunk_bounds,
                     excerpts,
                     data_chunk,
                     get_excerpts,
                     PerClusterData,
                     _range_from_slice,
                     _pad,
                     _load_arrays,
                     _save_arrays,
                     )
from ..testing import _assert_equal as ae
from ...io.mock import artificial_spike_clusters


#------------------------------------------------------------------------------
# Test utility functions
#------------------------------------------------------------------------------

def test_range_from_slice():
    """Test '_range_from_slice'."""

    class _SliceTest(object):
        """Utility class to make it more convenient to test slice objects."""
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        def __getitem__(self, item):
            if isinstance(item, slice):
                return _range_from_slice(item, **self._kwargs)

    with raises(ValueError):
        _SliceTest()[:]
    with raises(ValueError):
        _SliceTest()[1:]
    ae(_SliceTest()[:5], [0, 1, 2, 3, 4])
    ae(_SliceTest()[1:5], [1, 2, 3, 4])

    with raises(ValueError):
        _SliceTest()[::2]
    with raises(ValueError):
        _SliceTest()[1::2]
    ae(_SliceTest()[1:5:2], [1, 3])

    with raises(ValueError):
        _SliceTest(start=0)[:]
    with raises(ValueError):
        _SliceTest(start=1)[:]
    with raises(ValueError):
        _SliceTest(step=2)[:]

    ae(_SliceTest(stop=5)[:], [0, 1, 2, 3, 4])
    ae(_SliceTest(start=1, stop=5)[:], [1, 2, 3, 4])
    ae(_SliceTest(stop=5)[1:], [1, 2, 3, 4])
    ae(_SliceTest(start=1)[:5], [1, 2, 3, 4])
    ae(_SliceTest(start=1, step=2)[:5], [1, 3])
    ae(_SliceTest(start=1)[:5:2], [1, 3])

    ae(_SliceTest(length=5)[:], [0, 1, 2, 3, 4])
    with raises(ValueError):
        _SliceTest(length=5)[:3]
    ae(_SliceTest(length=5)[:10], [0, 1, 2, 3, 4])
    ae(_SliceTest(length=5)[:5], [0, 1, 2, 3, 4])
    ae(_SliceTest(start=1, length=5)[:], [1, 2, 3, 4, 5])
    ae(_SliceTest(start=1, length=5)[:6], [1, 2, 3, 4, 5])
    with raises(ValueError):
        _SliceTest(start=1, length=5)[:4]
    ae(_SliceTest(start=1, step=2, stop=5)[:], [1, 3])
    ae(_SliceTest(start=1, stop=5)[::2], [1, 3])
    ae(_SliceTest(stop=5)[1::2], [1, 3])


def test_pad():
    arr = np.random.rand(10, 3)

    ae(_pad(arr, 0, 'right'), arr[:0, :])
    ae(_pad(arr, 3, 'right'), arr[:3, :])
    ae(_pad(arr, 9), arr[:9, :])
    ae(_pad(arr, 10), arr)

    ae(_pad(arr, 12, 'right')[:10, :], arr)
    ae(_pad(arr, 12)[10:, :], np.zeros((2, 3)))

    ae(_pad(arr, 0, 'left'), arr[:0, :])
    ae(_pad(arr, 3, 'left'), arr[7:, :])
    ae(_pad(arr, 9, 'left'), arr[1:, :])
    ae(_pad(arr, 10, 'left'), arr)

    ae(_pad(arr, 12, 'left')[2:, :], arr)
    ae(_pad(arr, 12, 'left')[:2, :], np.zeros((2, 3)))

    with raises(ValueError):
        _pad(arr, -1)


def test_unique():
    """Test _unique() function"""
    _unique([])

    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    ae(_unique(spike_clusters), np.arange(n_clusters))


def test_normalize():
    """Test _normalize() function."""

    n_channels = 10
    positions = 1 + 2 * np.random.randn(n_channels, 2)

    # Keep ration is False.
    positions_n = _normalize(positions)

    x_min, y_min = positions_n.min(axis=0)
    x_max, y_max = positions_n.max(axis=0)

    np.allclose(x_min, 0.)
    np.allclose(x_max, 1.)
    np.allclose(y_min, 0.)
    np.allclose(y_max, 1.)

    # Keep ratio is True.
    positions_n = _normalize(positions, keep_ratio=True)

    x_min, y_min = positions_n.min(axis=0)
    x_max, y_max = positions_n.max(axis=0)

    np.allclose(min(x_min, y_min), 0.)
    np.allclose(max(x_max, y_max), 1.)
    np.allclose(x_min + x_max, 1)
    np.allclose(y_min + y_max, 1)


def test_index_of():
    """Test _index_of."""
    arr = [36, 42, 42, 36, 36, 2, 42]
    lookup = _unique(arr)
    ae(_index_of(arr, lookup), [1, 2, 2, 1, 1, 0, 2])


def test_as_array():
    ae(_as_array(3), [3])
    ae(_as_array([3]), [3])
    ae(_as_array(3.), [3.])
    ae(_as_array([3.]), [3.])

    with raises(ValueError):
        _as_array(map)


def test_in_polygon():
    polygon = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    points = np.random.uniform(size=(100, 2), low=-1, high=1)
    idx_expected = np.nonzero((points[:, 0] > 0) &
                              (points[:, 1] > 0) &
                              (points[:, 0] < 1) &
                              (points[:, 1] < 1))[0]
    idx = np.nonzero(_in_polygon(points, polygon))[0]
    ae(idx, idx_expected)


#------------------------------------------------------------------------------
# Test I/O functions
#------------------------------------------------------------------------------

@mark.parametrize('n', [20, 0])
def test_load_save_arrays(tempdir, n):
    path = op.join(tempdir, 'test.npy')
    # Random arrays.
    arrays = []
    for i in range(n):
        size = np.random.randint(low=3, high=50)
        assert size > 0
        arr = np.random.rand(size, 3).astype(np.float32)
        arrays.append(arr)
    _save_arrays(path, arrays)

    arrays_loaded = _load_arrays(path)

    assert len(arrays) == len(arrays_loaded)
    for arr, arr_loaded in zip(arrays, arrays_loaded):
        assert arr.shape == arr_loaded.shape
        assert arr.dtype == arr_loaded.dtype
        ae(arr, arr_loaded)


#------------------------------------------------------------------------------
# Test chunking
#------------------------------------------------------------------------------

def test_chunk_bounds():
    chunks = chunk_bounds(200, 100, overlap=20)

    assert next(chunks) == (0, 100, 0, 90)
    assert next(chunks) == (80, 180, 90, 170)
    assert next(chunks) == (160, 200, 170, 200)


def test_chunk():
    data = np.random.randn(200, 4)
    chunks = chunk_bounds(data.shape[0], 100, overlap=20)

    with raises(ValueError):
        data_chunk(data, (0, 0, 0))

    assert data_chunk(data, (0, 0)).shape == (0, 4)

    # Chunk 1.
    ch = next(chunks)
    d = data_chunk(data, ch)
    d_o = data_chunk(data, ch, with_overlap=True)

    ae(d_o, data[0:100])
    ae(d, data[0:90])

    # Chunk 2.
    ch = next(chunks)
    d = data_chunk(data, ch)
    d_o = data_chunk(data, ch, with_overlap=True)

    ae(d_o, data[80:180])
    ae(d, data[90:170])


def test_excerpts_1():
    bounds = [(start, end) for (start, end) in excerpts(100,
                                                        n_excerpts=3,
                                                        excerpt_size=10)]
    assert bounds == [(0, 10), (45, 55), (90, 100)]


def test_excerpts_2():
    bounds = [(start, end) for (start, end) in excerpts(10,
                                                        n_excerpts=3,
                                                        excerpt_size=10)]
    assert bounds == [(0, 10)]


def test_get_excerpts():
    data = np.random.rand(100, 2)
    subdata = get_excerpts(data, n_excerpts=10, excerpt_size=5)
    assert subdata.shape == (50, 2)
    ae(subdata[:5, :], data[:5, :])
    ae(subdata[-5:, :], data[-10:-5, :])

    data = np.random.rand(10, 2)
    subdata = get_excerpts(data, n_excerpts=10, excerpt_size=5)
    ae(subdata, data)

    data = np.random.rand(10, 2)
    subdata = get_excerpts(data, n_excerpts=1, excerpt_size=10)
    ae(subdata, data)


#------------------------------------------------------------------------------
# Test spike clusters functions
#------------------------------------------------------------------------------

def test_spikes_in_clusters():
    """Test _spikes_in_clusters()."""

    n_spikes = 1000
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    ae(_spikes_in_clusters(spike_clusters, []), [])

    for i in range(n_clusters):
        assert np.all(spike_clusters[_spikes_in_clusters(spike_clusters,
                                                         [i])] == i)

    clusters = [1, 5, 9]
    assert np.all(np.in1d(spike_clusters[_spikes_in_clusters(spike_clusters,
                                                             clusters)],
                          clusters))


def test_spikes_per_cluster():
    """Test _spikes_per_cluster()."""

    n_spikes = 1000
    spike_ids = np.arange(n_spikes).astype(np.int64)
    n_clusters = 10
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    spikes_per_cluster = _spikes_per_cluster(spike_ids, spike_clusters)
    assert list(spikes_per_cluster.keys()) == list(range(n_clusters))

    for i in range(10):
        ae(spikes_per_cluster[i], np.sort(spikes_per_cluster[i]))
        assert np.all(spike_clusters[spikes_per_cluster[i]] == i)

    sc = _flatten_spikes_per_cluster(spikes_per_cluster)
    ae(spike_clusters, sc)


def test_concatenate_per_cluster_arrays():
    """Test _spikes_per_cluster()."""

    def _column(arr):
        out = np.zeros((len(arr), 10))
        out[:, 0] = arr
        return out

    # 8, 11, 12, 13, 17, 18, 20
    spikes_per_cluster = {2: [11, 13, 17], 3: [8, 12], 5: [18, 20]}

    arrays_1d = {2: [1, 3, 7], 3: [8, 2], 5: [8, 0]}

    arrays_2d = {2: _column([1, 3, 7]),
                 3: _column([8, 2]),
                 5: _column([8, 0])}

    concat = _concatenate_per_cluster_arrays(spikes_per_cluster, arrays_1d)
    ae(concat, [8, 1, 2, 3, 7, 8, 0])

    concat = _concatenate_per_cluster_arrays(spikes_per_cluster, arrays_2d)
    ae(concat[:, 0], [8, 1, 2, 3, 7, 8, 0])
    ae(concat[:, 1:], np.zeros((7, 9)))


def test_per_cluster_data():

    spike_ids = [8, 11, 12, 13, 17, 18, 20]
    spc = {
        2: [11, 13, 17],
        3: [8, 12],
        5: [18, 20],
    }
    spike_clusters = [3, 2, 3, 2, 2, 5, 5]
    arrays = {
        2: [1, 3, 7],
        3: [8, 2],
        5: [8, 0],
    }
    array = [8, 1, 2, 3, 7, 8, 0]

    def _check(pcd):
        ae(pcd.spike_ids, spike_ids)
        ae(pcd.spike_clusters, spike_clusters)
        ae(pcd.array, array)
        ae(pcd.spc, spc)
        ae(pcd.arrays, arrays)

        # Check subset on 1 cluster.
        pcd_s = pcd.subset(clusters=[2])
        ae(pcd_s.spike_ids, [11, 13, 17])
        ae(pcd_s.spike_clusters, [2, 2, 2])
        ae(pcd_s.array, [1, 3, 7])
        ae(pcd_s.spc, {2: [11, 13, 17]})
        ae(pcd_s.arrays, {2: [1, 3, 7]})

        # Check subset on some spikes.
        pcd_s = pcd.subset(spike_ids=[11, 12, 13, 17])
        ae(pcd_s.spike_ids, [11, 12, 13, 17])
        ae(pcd_s.spike_clusters, [2, 3, 2, 2])
        ae(pcd_s.array, [1, 2, 3, 7])
        ae(pcd_s.spc, {2: [11, 13, 17], 3: [12]})
        ae(pcd_s.arrays, {2: [1, 3, 7], 3: [2]})

        # Check subset on 2 complete clusters.
        pcd_s = pcd.subset(clusters=[3, 5])
        ae(pcd_s.spike_ids, [8, 12, 18, 20])
        ae(pcd_s.spike_clusters, [3, 3, 5, 5])
        ae(pcd_s.array, [8, 2, 8, 0])
        ae(pcd_s.spc, {3: [8, 12], 5: [18, 20]})
        ae(pcd_s.arrays, {3: [8, 2], 5: [8, 0]})

        # Check subset on 2 incomplete clusters.
        pcd_s = pcd.subset(spc={3: [8, 12], 5: [20]})
        ae(pcd_s.spike_ids, [8, 12, 20])
        ae(pcd_s.spike_clusters, [3, 3, 5])
        ae(pcd_s.array, [8, 2, 0])
        ae(pcd_s.spc, {3: [8, 12], 5: [20]})
        ae(pcd_s.arrays, {3: [8, 2], 5: [0]})

    # From flat arrays.
    pcd = PerClusterData(spike_ids=spike_ids,
                         array=array,
                         spike_clusters=spike_clusters,
                         )
    _check(pcd)

    # From dicts.
    pcd = PerClusterData(spc=spc, arrays=arrays)
    _check(pcd)
