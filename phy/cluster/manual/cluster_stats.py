# -*- coding: utf-8 -*-

"""Cluster statistics."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from .cluster_metadata import BaseClusterInfo


#------------------------------------------------------------------------------
# ClusterStats class
#------------------------------------------------------------------------------

class ClusterStats(BaseClusterInfo):
    def __init__(self, fields=None):
        super(ClusterStats, self).__init__(fields=fields)
