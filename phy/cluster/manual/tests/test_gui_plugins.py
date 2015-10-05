# -*- coding: utf-8 -*-

"""Test GUI plugins."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from phy.gui.tests.test_gui import gui


#------------------------------------------------------------------------------
# Test GUI plugins
#------------------------------------------------------------------------------

def test_manual_clustering(qtbot, gui, spike_clusters, cluster_metadata):
    gui.attach('ManualClustering',
               spike_clusters=spike_clusters,
               cluster_metadata=cluster_metadata,
               )
