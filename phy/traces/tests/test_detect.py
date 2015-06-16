
# -*- coding: utf-8 -*-

"""Tests of spike detection routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae

from ..detect import Thresholder, connected_components
from ...io.mock import artificial_traces


#------------------------------------------------------------------------------
# Test thresholder
#------------------------------------------------------------------------------

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


#------------------------------------------------------------------------------
# Test connected components
#------------------------------------------------------------------------------

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

    if not isinstance(chunk, np.ndarray):
        chunk = np.array(chunk)
    strong_crossings = kwargs.pop('strong_crossings', None)
    if strong_crossings is not None and not isinstance(strong_crossings, np.ndarray):
        strong_crossings = np.array(strong_crossings)

    comp = connected_components(chunk,
                                probe_adjacency_list=probe_adjacency_list,
                                strong_crossings=strong_crossings,
                                **kwargs)
    assert len(comp) == len(components), (len(comp), len(components))
    for c1, c2 in zip(comp, components):
        assert set(c1) == set(c2)


def test_components():
    # 1 time step, 1 element
    _test_components([[0, 0, 0, 0, 0]],  [])

    _test_components([[1, 0, 0, 0, 0]],  [[(0, 0)]])

    _test_components([
            [0, 1, 0, 0, 0],
        ],  [[(0, 1)]])

    _test_components([
            [0, 0, 0, 1, 0],
        ],  [[(0, 3)]])

    _test_components([
            [0, 0, 0, 0, 1],
        ],  [[(0, 4)]])

    # 1 time step, 2 elements
    _test_components([
            [1, 1, 0, 0, 0],
        ],  [[(0, 0), (0, 1)]])

    _test_components([
            [1, 0, 1, 0, 0],
        ],  [[(0, 0)], [(0, 2)]])

    _test_components([
            [1, 0, 0, 0, 1],
        ],  [[(0, 0)], [(0, 4)]])

    _test_components([
            [0, 1, 0, 1, 0],
        ],  [[(0, 1)], [(0, 3)]])

    # 1 time step, 3 elements
    _test_components([
            [1, 1, 1, 0, 0],
        ],  [[(0, 0), (0, 1), (0, 2)]])

    _test_components([
            [1, 1, 0, 1, 0],
        ],  [[(0, 0), (0, 1)], [(0, 3)]])

    _test_components([
            [1, 0, 1, 1, 0],
        ],  [[(0, 0)], [(0, 2), (0, 3)]])

    _test_components([
            [0, 1, 1, 1, 0],
        ],  [[(0, 1), (0, 2), (0, 3)]])

    _test_components([
            [0, 1, 1, 0, 1],
        ],  [[(0, 1), (0, 2)], [(0, 4)]])


    # 5 time steps, varying join_size
    _test_components([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
        ],  [[(1, 2)],
             [(2, 0)], [(2, 2), (2, 3)],
             [(3, 0)], [(3, 3)],
             [(4, 1)], [(4, 3), (4, 4)],
             ])

    _test_components([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
        ],  [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4)],
             [(2, 0), (3, 0), (4, 1)],
             ],
        join_size=1
        )

    _test_components(None,
            [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4),
              (2, 0), (3, 0), (4, 1)],
             ],
        join_size=2
        )

    # 5 time steps, strong != weak
    _test_components(None,
            [],
        join_size=0,
        strong_crossings=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        )

    _test_components(None,
            [[(1, 2)],
             ],
        join_size=0,
        strong_crossings=[
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        )

    _test_components(None,
            [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4)],
             ],
        join_size=1,
        strong_crossings=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ]
        )

    _test_components(None,
            [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4),
              (2, 0), (3, 0), (4, 1)],
             ],
        join_size=2,
        strong_crossings=[
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        )

    _test_components(None,
            [[(1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4),
              (2, 0), (3, 0), (4, 1)],
             ],
        join_size=2,
        strong_crossings=[
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]
        )
