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
    """Hold cluster statistics with cache.

    Initialized as:

        ClustersStats(my_stat=my_function)

    where `my_function(cluster)` returns a cluster statistics.

    ClusterStats handles the caching logic. It provides an
    `invalidate(clusters)` method.

    """
    def __init__(self, **functions):
        # Set the methods.
        for name, fun in functions.items():
            setattr(self, name, lambda cluster: self.get(cluster, name))
        super(ClusterStats, self).__init__(fields=functions)

    def invalidate(self, clusters):
        """Invalidate clusters from the cache."""
        self.unset(clusters)
