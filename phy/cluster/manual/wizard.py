# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from operator import itemgetter

import numpy as np


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _argsort(seq, reverse=True, n_max=None):
    """Return the list of clusters in decreasing order of value from
    a list of tuples (cluster, value)."""
    out = [cl for (cl, v) in sorted(seq, key=itemgetter(1),
                                    reverse=reverse)]
    if n_max in (None, 0):
        return out
    else:
        return out[:n_max]


def _best_clusters(clusters, quality, n_max=None):
    return _argsort([(cluster, quality(cluster))
                     for cluster in clusters], n_max=n_max)


#------------------------------------------------------------------------------
# Wizard
#------------------------------------------------------------------------------

class Wizard(object):
    def __init__(self, cluster_ids=None):
        self._similarity = None
        self._quality = None
        self._ignored = set()
        self.cluster_ids = cluster_ids

    @property
    def cluster_ids(self):
        return self._cluster_ids

    @cluster_ids.setter
    def cluster_ids(self, cluster_ids):
        if isinstance(cluster_ids, np.ndarray):
            cluster_ids = cluster_ids.tolist()
        self._cluster_ids = sorted(cluster_ids)

    def set_similarity(self, func):
        """Register a function returing the similarity between two clusters."""
        self._similarity = func
        return func

    def set_quality(self, func):
        """Register a function returing the quality of a cluster."""
        self._quality = func
        return func

    def _check_cluster_ids(self):
        if self._cluster_ids is None:
            raise RuntimeError("The list of clusters need to be set.")

    def _filter(self, items):
        """Filter out ignored clusters or pairs of clusters."""
        return [item for item in items
                if item not in self._ignored]

    # Public methods
    #--------------------------------------------------------------------------

    def best_clusters(self, n_max=10):
        """Return the list of best clusters sorted by decreasing quality."""
        self._check_cluster_ids()
        return self._filter(_best_clusters(self._cluster_ids, self._quality,
                                           n_max=n_max))

    def best_cluster(self):
        """Return the best cluster."""
        clusters = self.best_clusters(n_max=1)
        if clusters:
            return clusters[0]

    def most_similar_clusters(self, cluster=None, n_max=10):
        """Return the `n_max` most similar clusters."""
        if cluster is None:
            cluster = self.best_cluster()
        self._check_cluster_ids()
        similarity = [(other, self._similarity(cluster, other))
                      for other in self._cluster_ids
                      if other != cluster]
        clusters = _argsort(similarity, n_max=n_max)
        pairs = zip([cluster] * len(clusters), clusters)
        pairs = self._filter(pairs)
        return [clu for (_, clu) in pairs]

    def ignore(self, cluster_or_pair):
        """Mark a cluster or a pair of clusters as ignored."""
        self._ignored.add(cluster_or_pair)
