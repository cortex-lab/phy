# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from phy.io.store import get_closest_clusters


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def cluster_ids():
    yield [0, 1, 2, 10, 11, 20, 30]
    #      i, g, N,  i,  g,  N, N


@yield_fixture
def cluster_groups():
    yield {0: 'noise', 1: 'good', 10: 'mua', 11: 'good'}


@yield_fixture
def quality():
    yield lambda c: c


@yield_fixture
def similarity(cluster_ids):
    sim = lambda c, d: (c * 1.01 + d)
    yield lambda c: get_closest_clusters(c, cluster_ids, sim)
