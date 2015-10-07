# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from ..wizard import Wizard, _wizard_group, best_quality_strategy
from .._utils import create_cluster_meta


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def cluster_groups():
    data = {1: 'noise',
            2: 'mua',
            11: 'good',
            12: 'good',
            13: 'good',
            101: None,
            102: None,
            103: None,
            104: None,
            105: None,
            }
    yield data


@yield_fixture
def cluster_meta(cluster_groups):
    yield create_cluster_meta(cluster_groups)


@yield_fixture
def mock_wizard():

    wizard = Wizard()
    wizard.set_cluster_ids_function(lambda: [1, 2, 3])

    @wizard.set_quality_function
    def quality(cluster):
        return cluster

    @wizard.set_similarity_function
    def similarity(cluster, other):
        return cluster + other

    yield wizard


@yield_fixture
def wizard_with_groups(mock_wizard, cluster_groups):

    def get_cluster_ids():
        return sorted(cluster_groups.keys())

    mock_wizard.set_cluster_ids_function(get_cluster_ids)

    @mock_wizard.set_status_function
    def status(cluster):
        group = cluster_groups.get(cluster, None)
        return _wizard_group(group)

    mock_wizard.set_strategy_function(best_quality_strategy)

    yield mock_wizard
