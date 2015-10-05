# -*- coding: utf-8 -*-

"""Test GUI plugins."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

# from pytest

from .test_wizard import clustering, cluster_metadata, wizard  # noqa
from phy.gui.tests.test_gui import gui  # noqa


#------------------------------------------------------------------------------
# Test GUI plugins
#------------------------------------------------------------------------------

def test_manual_clustering(qtbot, gui, clustering, cluster_metadata):  # noqa
    # TODO: refactor these fixtures
    sc = clustering.spike_clusters
    gui.attach('ManualClustering',
               spike_clusters=sc,
               cluster_metadata=cluster_metadata._cluster_metadata,
               )
