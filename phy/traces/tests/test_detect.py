# -*- coding: utf-8 -*-

"""Tests of spike detection routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac

from ..detect import (compute_threshold,
                      Thresholder,
                      connected_components,
                      FloodFillDetector,
                      )
from ...io.mock import artificial_traces


#------------------------------------------------------------------------------
# Test thresholder
#------------------------------------------------------------------------------

def test_compute_threshold():
    n_samples, n_channels = 100, 10
    data = artificial_traces(n_samples, n_channels)

    # Single threshold.
    threshold = compute_threshold(data, std_factor=1.)
    assert threshold.shape == (2,)
    assert threshold[0] > 0
    assert threshold[0] == threshold[1]

    threshold = compute_threshold(data, std_factor=[1., 2.])
    assert threshold.shape == (2,)
    assert threshold[1] == 2 * threshold[0]

    # Multiple threshold.
    threshold = compute_threshold(data, single_threshold=False, std_factor=2.)
    assert threshold.shape == (2, n_channels)

    threshold = compute_threshold(data,
                                  single_threshold=False,
                                  std_factor=(1., 2.))
    assert threshold.shape == (2, n_channels)
    ac(threshold[1], 2 * threshold[0])


def test_thresholder():
    n_samples, n_channels = 100, 12
    strong, weak = .1, .2

    data = artificial_traces(n_samples, n_channels)

    # Positive and strong.
    thresholder = Thresholder(mode='positive',
                              thresholds=strong)
    ae(thresholder(data), data > strong)

    # Negative and weak.
    thresholder = Thresholder(mode='negative',
                              thresholds={'weak': weak})
    ae(thresholder(data), data < -weak)

    # Both and strong+weak.
    thresholder = Thresholder(mode='both',
                              thresholds={'weak': weak,
                                          'strong': strong,
                                          })
    ae(thresholder(data, 'weak'), np.abs(data) > weak)
    ae(thresholder(data, threshold='strong'), np.abs(data) > strong)

    # Multiple thresholds.
    t = thresholder(data, ('weak', 'strong'))
    ae(t['weak'], np.abs(data) > weak)
    ae(t['strong'], np.abs(data) > strong)

    # Array threshold.
    thre = np.linspace(weak - .05, strong + .05, n_channels)
    thresholder = Thresholder(mode='positive', thresholds=thre)
    t = thresholder(data)
    assert t.shape == data.shape
    ae(t, data > thre)


#------------------------------------------------------------------------------
# Test connected components
#------------------------------------------------------------------------------

def _as_set(c):
    if isinstance(c, np.ndarray):
        c = c.tolist()
    c = [tuple(_) for _ in c]
    return set(c)


def _assert_components_equal(cc1, cc2):
    assert len(cc1) == len(cc2)
    for c1, c2 in zip(cc1, cc2):
        assert _as_set(c1) == _as_set(c2)


def _test_components(chunk=None, components=None, **kwargs):

    def _clip(x, m, M):
        return [_ for _ in x if m <= _ < M]

    n = 5
    probe_adjacency_list = {i: set(_clip([i - 1, i + 1], 0, n))
                            for i in range(n)}

    if chunk is None:
        chunk = [[0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [1, 0, 1, 1, 0],
                 [1, 0, 0, 1, 0],
                 [0, 1, 0, 1, 1],
                 ]

    if components is None:
        components = []

    if not isinstance(chunk, np.ndarray):
        chunk = np.array(chunk)
    strong_crossings = kwargs.pop('strong_crossings', None)
    if (strong_crossings is not None and
            not isinstance(strong_crossings, np.ndarray)):
        strong_crossings = np.array(strong_crossings)

    comp = connected_components(chunk,
                                probe_adjacency_list=probe_adjacency_list,
                                strong_crossings=strong_crossings,
                                **kwargs)
    _assert_components_equal(comp, components)


def test_components():
    # 1 time step, 1 element
    _test_components([[0, 0, 0, 0, 0]], [])

    _test_components([[1, 0, 0, 0, 0]], [[(0, 0)]])

    _test_components([[0, 1, 0, 0, 0]], [[(0, 1)]])

    _test_components([[0, 0, 0, 1, 0]], [[(0, 3)]])

    _test_components([[0, 0, 0, 0, 1]], [[(0, 4)]])

    # 1 time step, 2 elements
    _test_components([[1, 1, 0, 0, 0]], [[(0, 0), (0, 1)]])

    _test_components([[1, 0, 1, 0, 0]], [[(0, 0)], [(0, 2)]])

    _test_components([[1, 0, 0, 0, 1]], [[(0, 0)], [(0, 4)]])

    _test_components([[0, 1, 0, 1, 0]], [[(0, 1)], [(0, 3)]])

    # 1 time step, 3 elements
    _test_components([[1, 1, 1, 0, 0]], [[(0, 0), (0, 1), (0, 2)]])

    _test_components([[1, 1, 0, 1, 0]], [[(0, 0), (0, 1)], [(0, 3)]])

    _test_components([[1, 0, 1, 1, 0]], [[(0, 0)], [(0, 2), (0, 3)]])

    _test_components([[0, 1, 1, 1, 0]], [[(0, 1), (0, 2), (0, 3)]])

    _test_components([[0, 1, 1, 0, 1]], [[(0, 1), (0, 2)], [(0, 4)]])

    # 5 time steps, varying join_size
    _test_components([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1],
    ], [[(1, 2)],
        [(2, 0)],
        [(2, 2), (2, 3)],
        [(3, 0)],
        [(3, 3)],
        [(4, 1)],
        [(4, 3), (4, 4)],
        ])

    _test_components([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1],
    ], [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4)],
        [(2, 0), (3, 0), (4, 1)]], join_size=1)

    _test_components(
        components=[[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4),
                     (2, 0), (3, 0), (4, 1)]], join_size=2)

    # 5 time steps, strong != weak
    _test_components(join_size=0,
                     strong_crossings=[
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])

    _test_components(components=[[(1, 2)]],
                     join_size=0,
                     strong_crossings=[
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])

    _test_components(
        components=[[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4)]],
        join_size=1,
        strong_crossings=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1]])

    _test_components(
        components=[[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4),
                     (2, 0), (3, 0), (4, 1)]],
        join_size=2,
        strong_crossings=[
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])

    _test_components(
        components=[[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4),
                     (2, 0), (3, 0), (4, 1)]],
        join_size=2,
        strong_crossings=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]])


def test_flood_fill():

    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: []}

    ff = FloodFillDetector(probe_adjacency_list=graph,
                           join_size=1,
                           )

    weak = [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            ]

    # Weak - weak
    comps = [[(1, 1), (1, 2)],
             [(3, 2), (4, 2)],
             [(4, 3)],
             [(6, 3)],
             ]
    cc = ff(weak, weak)
    _assert_components_equal(cc, comps)

    # Weak and strong
    strong = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0],
              ]

    comps = [[(3, 2), (4, 2)],
             [(4, 3)],
             [(6, 3)],
             ]
    cc = ff(weak, strong)
    _assert_components_equal(cc, comps)
