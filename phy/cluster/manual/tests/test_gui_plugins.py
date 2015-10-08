# -*- coding: utf-8 -*-

"""Test GUI plugins."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture
import numpy as np

from phy.gui.tests.test_gui import gui  # noqa


#------------------------------------------------------------------------------
# Test GUI plugins
#------------------------------------------------------------------------------

@yield_fixture  # noqa
def manual_clustering(qtbot, gui, cluster_ids, cluster_groups):
    spike_clusters = np.array(cluster_ids)

    mc = gui.attach('ManualClustering',
                    spike_clusters=spike_clusters,
                    cluster_groups=cluster_groups,
                    )

    _s = []

    # Connect to the `select` event.
    @mc.gui.connect_
    def on_select(cluster_ids, spike_ids):
        _s.append((cluster_ids, spike_ids))

    def assert_selection(*cluster_ids):  # pragma: no cover
        assert _s[-1][0] == list(cluster_ids)
        if len(cluster_ids) >= 1:
            assert mc.wizard.best == cluster_ids[0]
        elif len(cluster_ids) >= 2:
            assert mc.wizard.match == cluster_ids[2]

    yield mc, assert_selection


def test_manual_clustering_1(manual_clustering):
    mc, ae = manual_clustering
    print(mc.wizard.selection)
