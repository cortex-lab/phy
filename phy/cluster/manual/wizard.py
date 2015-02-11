# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# Wizard
#------------------------------------------------------------------------------

class Wizard(object):
    def __init__(self, cluster_stats, cluster_metadata):
        pass

    def best_clusters(self, n_max=None):  # decreasing order of quality
        pass

    def best_cluster(self):
        pass

    def most_similar_clusters(self, cluster, n_max=None):
        pass

    def mark_dissimilar(self, clusters):
        pass
