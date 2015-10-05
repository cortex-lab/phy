# -*- coding: utf-8 -*-

"""Test GUI plugins."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture
from numpy.testing import assert_array_equal as ae

from .conftest import _set_test_wizard
from phy.gui.tests.test_gui import gui  # noqa


#------------------------------------------------------------------------------
# Test GUI plugins
#------------------------------------------------------------------------------

@yield_fixture
def manual_clustering(qtbot, gui, spike_clusters,  # noqa
                      cluster_metadata):
    mc = gui.attach('ManualClustering',
                    spike_clusters=spike_clusters,
                    cluster_metadata=cluster_metadata,
                    )
    _set_test_wizard(mc.wizard)
    yield mc


def test_manual_clustering(manual_clustering):
    actions = manual_clustering.actions
    wizard = manual_clustering.wizard

    # Test cluster ids.
    ae(manual_clustering.cluster_ids, [2, 3, 5, 7])

    # Connect to the `select` event.
    _s = []

    @manual_clustering.gui.connect_
    def on_select(cluster_ids, spike_ids):
        _s.append((cluster_ids, spike_ids))

    # Test select actions.
    actions.select([])
    ae(_s[-1][0], [])
    ae(_s[-1][1], [])

    # Test wizard actions.
    actions.reset_wizard()
    assert wizard.best_list == [3, 2, 7, 5]
    assert wizard.best == 3

    actions.next()
    assert wizard.best == 2

    actions.last()
    assert wizard.best == 5

    actions.previous()
    assert wizard.best == 7

    actions.first()
    assert wizard.best == 3

    # Test pinning.
    actions.pin()
    assert wizard.match_list == [2, 7, 5]
    assert wizard.match == 2
    wizard.next()
    assert wizard.match == 7
    assert len(_s) == 9
