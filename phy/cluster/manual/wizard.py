# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import math
from operator import itemgetter

import numpy as np


#------------------------------------------------------------------------------
# Wizard
#------------------------------------------------------------------------------

def _norm(x):
    """Euclidean norm of a vector."""
    return math.sqrt((x ** 2).sum())


def _cluster_masks_similarity(masks_ref, masks):
    """Compute the similarity between two cluster masks."""
    return np.dot(masks_ref / _norm(masks_ref),
                  masks / _norm(masks))


class Wizard(object):
    def __init__(self, cluster_stats=None, cluster_metadata=None):
        self._cluster_stats = cluster_stats
        self._cluster_metadata = cluster_metadata

    @property
    def _clusters(self):
        """Return the sorted list of valid clusters."""
        clusters = self._cluster_stats.keys()
        # Checking the keys in both dictionaries... probably not necessary.
        # clusters_ = self._cluster_metadata.keys()
        # assert clusters == clusters_
        return clusters

    def best_clusters(self, n_max=None):
        """Return the list of best clusters sorted by decreasing quality."""
        quality = [(cluster, self._cluster_stats[cluster]['quality'])
                   for cluster in self._clusters]
        quality_s = sorted(quality, key=itemgetter(1), reverse=True)
        return [cluster for cluster, qual in quality_s[:n_max]]

    def best_cluster(self):
        """Return the best cluster."""
        clusters = self.best_clusters(n_max=1)
        if clusters:
            return clusters[0]

    def most_similar_clusters(self, cluster, n_max=None):
        """Return the `n_max` most similar clusters."""
        masks_ref = self._cluster_stats[cluster]['cluster_masks']
        # TODO: select the appropriate groups
        masks = [(other, self._cluster_stats[other]['cluster_masks'])
                 for other in self._clusters
                 if other != cluster]
        similarity = [(clu, _cluster_masks_similarity(masks_ref, m))
                      for (clu, m) in masks]
        # TODO: refactor this snippet with best_clusters()
        similarity_s = sorted(similarity, key=itemgetter(1), reverse=True)
        return [clu for clu, sim in similarity_s[:n_max]]

    def mark_dissimilar(self, cluster_0, cluster_1):
        """Mark two clusters as dissimilar after a human decision.

        This pair should not be reproposed again to the user.

        """
        # TODO
        pass
