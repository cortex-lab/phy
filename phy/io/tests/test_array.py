# -*- coding: utf-8 -*-

"""Tests of array utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from pytest import raises

from ..array import (_unique,
                     _normalize,
                     _index_of,
                     _in_polygon,
                     _spikes_in_clusters,
                     _spikes_per_cluster,
                     _flatten_per_cluster,
                     _get_data_lim,
                     select_spikes,
                     Selector,
                     chunk_bounds,
                     regular_subset,
                     excerpts,
                     data_chunk,
                     grouped_mean,
                     get_excerpts,
                     _concatenate_virtual_arrays,
                     _range_from_slice,
                     _pad,
                     _get_padded,
                     read_array,
                     write_array,
                     )
from phy.utils._types import _as_array
from phy.utils.testing import _assert_equal as ae
from ..mock import artificial_spike_clusters


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


def test_get_padded():
    arr = np.array([1, 2, 3])[:, np.newaxis]

    with raises(RuntimeError):
        ae(_get_padded(arr, -2, 5).ravel(), [1, 2, 3, 0, 0])
    ae(_get_padded(arr, 1, 2).ravel(), [2])
    ae(_get_padded(arr, 0, 5).ravel(), [1, 2, 3, 0, 0])
    ae(_get_padded(arr, -2, 3).ravel(), [0, 0, 1, 2, 3])


def test_get_data_lim():
    arr = np.random.rand(10, 5)
    assert 0 < _get_data_lim(arr) < 1
    assert 0 < _get_data_lim(arr, 2) < 1


def test_unique():
    """Test _unique() function"""
    _unique([])

    n_spikes = 300
    n_clusters = 3
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
# Test read/save
#------------------------------------------------------------------------------

def test_read_write(tempdir):
    arr = np.arange(10).astype(np.float32)

    path = op.join(tempdir, 'test.npy')

    write_array(path, arr)
    ae(read_array(path), arr)
    ae(read_array(path, mmap_mode='r'), arr)


#------------------------------------------------------------------------------
# Test virtual concatenation
#------------------------------------------------------------------------------

def test_concatenate_virtual_arrays_1():
    arrs = [np.arange(5), np.arange(10, 12), np.array([0])]
    c = _concatenate_virtual_arrays(arrs, scaling=1)
    assert c.shape == (8,)
    assert c._get_recording(3) == 0
    assert c._get_recording(5) == 1

    ae(c[:], [0, 1, 2, 3, 4, 10, 11, 0])
    ae(c[0], [0])
    ae(c[4], [4])
    ae(c[5], [10])
    ae(c[6], [11])

    ae(c[4:6], [4, 10])

    ae(c[:6], [0, 1, 2, 3, 4, 10])
    ae(c[4:], [4, 10, 11, 0])
    ae(c[4:-1], [4, 10, 11])


def test_concatenate_virtual_arrays_2():
    arrs = [np.zeros((2, 2)), np.ones((3, 2))]
    c = _concatenate_virtual_arrays(arrs)
    assert c.shape == (5, 2)
    ae(c[:, :], np.vstack((np.zeros((2, 2)), np.ones((3, 2)))))
    ae(c[0:4, 0], [0, 0, 1, 1])


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

    assert len(get_excerpts(data, n_excerpts=0, excerpt_size=10)) == 0


def test_regular_subset():
    spikes = [2, 3, 5, 7, 11, 13, 17]
    ae(regular_subset(spikes), spikes)
    ae(regular_subset(spikes, 100), spikes)
    ae(regular_subset(spikes, 100, offset=2), spikes)
    ae(regular_subset(spikes, 3), [2, 7, 17])
    ae(regular_subset(spikes, 3, offset=1), [3, 11])


#------------------------------------------------------------------------------
# Test spike clusters functions
#------------------------------------------------------------------------------

def test_spikes_in_clusters():
    """Test _spikes_in_clusters()."""

    n_spikes = 100
    n_clusters = 5
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    ae(_spikes_in_clusters(spike_clusters, []), [])

    for i in range(n_clusters):
        assert np.all(spike_clusters[_spikes_in_clusters(spike_clusters,
                                                         [i])] == i)

    clusters = [1, 2, 3]
    assert np.all(np.in1d(spike_clusters[_spikes_in_clusters(spike_clusters,
                                                             clusters)],
                          clusters))


def test_spikes_per_cluster():
    """Test _spikes_per_cluster()."""

    n_spikes = 100
    n_clusters = 3
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    assert not _spikes_per_cluster([])

    spikes_per_cluster = _spikes_per_cluster(spike_clusters)
    assert list(spikes_per_cluster.keys()) == list(range(n_clusters))

    for i in range(n_clusters):
        ae(spikes_per_cluster[i], np.sort(spikes_per_cluster[i]))
        assert np.all(spike_clusters[spikes_per_cluster[i]] == i)


def test_flatten_per_cluster():
    spc = {2: [2, 7, 11], 3: [3, 5], 5: []}
    arr = _flatten_per_cluster(spc)
    ae(arr, [2, 3, 5, 7, 11])


def test_grouped_mean():
    spike_clusters = np.array([2, 3, 2, 2, 5])
    arr = spike_clusters * 10
    ae(grouped_mean(arr, spike_clusters), [20, 30, 50])


def test_select_spikes():
    with raises(AssertionError):
        select_spikes()
    spikes = [2, 3, 5, 7, 11]
    spc = lambda c: {2: [2, 7, 11], 3: [3, 5], 5: []}.get(c, None)
    ae(select_spikes([], spikes_per_cluster=spc), [])
    ae(select_spikes([2, 3, 5], spikes_per_cluster=spc), spikes)
    ae(select_spikes([2, 5], spikes_per_cluster=spc), spc(2))

    ae(select_spikes([2, 3, 5], 0, spikes_per_cluster=spc), spikes)
    ae(select_spikes([2, 3, 5], None, spikes_per_cluster=spc), spikes)
    ae(select_spikes([2, 3, 5], 1, spikes_per_cluster=spc), [2, 3])
    ae(select_spikes([2, 5], 2, spikes_per_cluster=spc), [2])

    sel = Selector(spc)
    assert sel.select_spikes() is None
    ae(sel.select_spikes([2, 5]), spc(2))
    ae(sel.select_spikes([2, 5], 2), [2])
