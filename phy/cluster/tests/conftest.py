# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture

from phy.io.array import (get_closest_clusters,
                          )


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def cluster_ids():
    return [0, 1, 2, 10, 11, 20, 30]
    #       i, g, N,  i,  g,  N, N


@fixture
def cluster_groups():
    return {0: 'noise', 1: 'good', 10: 'mua', 11: 'good'}


@fixture
def quality():
    def quality(c):
        return c
    return quality


@fixture
def similarity(cluster_ids):
    sim = lambda c, d: (c * 1.01 + d)

    def similarity(c):
        return get_closest_clusters(c, cluster_ids, sim)
    return similarity
