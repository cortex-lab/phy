# -*- coding: utf-8 -*-

"""Test GUI plugins."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from numpy.testing import assert_array_equal as ae

from phy.gui.tests.test_gui import gui  # noqa


#------------------------------------------------------------------------------
# Test GUI plugins
#------------------------------------------------------------------------------

def test_manual_clustering(qtbot, gui, spike_clusters,  # noqa
                           cluster_metadata):
    mc = gui.attach('ManualClustering',
                    spike_clusters=spike_clusters,
                    cluster_metadata=cluster_metadata,
                    )
    ae(mc.cluster_ids, [2, 3, 5, 7])
