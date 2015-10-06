# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from ..clustering import Clustering
from ..wizard import Wizard, _wizard_group
from .._utils import create_cluster_meta


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def spike_clusters():
    yield [2, 3, 5, 7]


@yield_fixture
def clustering(spike_clusters):
    yield Clustering(spike_clusters)


@yield_fixture
def cluster_groups():
    data = {2: None,
            3: None,
            5: 'mua',
            7: 'good',
            }
    yield data


@yield_fixture
def cluster_meta(cluster_groups):
    yield create_cluster_meta(cluster_groups)


def _set_test_wizard(wizard):

    @wizard.set_quality_function
    def quality(cluster):
        return cluster * .1

    @wizard.set_similarity_function
    def similarity(cluster, other):
        return 1. + quality(cluster) - quality(other)


@yield_fixture
def wizard(cluster_groups):
    wizard = Wizard()

    def get_cluster_ids():
        return [2, 3, 5, 7]

    wizard.set_cluster_ids_function(get_cluster_ids)

    @wizard.set_status_function
    def status(cluster):
        group = cluster_groups.get(cluster, None)
        return _wizard_group(group)

    _set_test_wizard(wizard)

    yield wizard
