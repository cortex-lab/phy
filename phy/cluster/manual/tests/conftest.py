# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from ..wizard import Wizard


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def cluster_ids():
    yield [0, 1, 2, 10, 20, 30]
    #      i, g, N, N,  N,  N


@yield_fixture
def get_cluster_ids(cluster_ids):
    yield lambda: cluster_ids


@yield_fixture
def cluster_groups():
    yield {0: 'ignored', 1: 'good'}


@yield_fixture
def status(cluster_groups):
    yield lambda c: cluster_groups.get(c, None)


@yield_fixture
def quality():
    yield lambda c: c


@yield_fixture
def similarity():
    yield lambda c, d: c * 1.01 + d


@yield_fixture
def wizard(get_cluster_ids, status, quality, similarity):
    wizard = Wizard()

    wizard.set_cluster_ids_function(get_cluster_ids)
    wizard.set_status_function(status)
    wizard.set_quality_function(quality)
    wizard.set_similarity_function(similarity)

    yield wizard
