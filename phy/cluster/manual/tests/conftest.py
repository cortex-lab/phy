# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from ..clustering import Clustering
from ..gui_plugins import create_cluster_metadata
from ..wizard import Wizard


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def cluster_ids():
    yield [2, 3, 5, 7]


@yield_fixture
def spike_clusters():
    yield [2, 3, 5, 7]


@yield_fixture
def clustering(spike_clusters):
    yield Clustering(spike_clusters)


@yield_fixture
def cluster_metadata():
    data = {2: {'group': 3},
            3: {'group': 3},
            5: {'group': 1},
            7: {'group': 2},
            }

    yield create_cluster_metadata(data)


@yield_fixture
def wizard():

    def get_cluster_ids():
        return [2, 3, 5, 7]

    wizard = Wizard()
    wizard.set_cluster_ids_function(get_cluster_ids)

    @wizard.set_status_function
    def cluster_status(cluster):
        return {2: None, 3: None, 5: 'ignored', 7: 'good'}.get(cluster, None)

    @wizard.set_quality_function
    def quality(cluster):
        return cluster * .1

    @wizard.set_similarity_function
    def similarity(cluster, other):
        return 1. + quality(cluster) - quality(other)

    yield wizard
