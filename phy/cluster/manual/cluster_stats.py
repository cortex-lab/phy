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
        # Set the method.
        for name, fun in functions.items():
            dec_fun = self._decorate(name, fun)
            setattr(self, name, lambda cluster: self.get(cluster, name))
        super(ClusterStats, self).__init__(fields=functions)

    def _decorate(self, name, fun):
        """Decorate a cluster ==> stat function into a defaultdict
        factory function."""
        def dec_fun(cluster):
            # Compute the statistic if it is not in cache.
            if cluster not in self._data:
                return self.set(cluster, name, fun(cluster))
            return self.get(cluster, name)
        return dec_fun

    def invalidate(self, clusters):
        """Invalidate clusters from the cache."""
        self.unset(clusters)
