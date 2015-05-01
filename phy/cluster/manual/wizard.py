# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from operator import itemgetter

import numpy as np

from ...ext.six import integer_types


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
    """Propose a selection of high-quality clusters and merge candidates."""
    def __init__(self, cluster_ids=None):
        self._similarity = None
        self._quality = None
        self._ignored = set()
        self.cluster_ids = cluster_ids

    # Internal methods
    #--------------------------------------------------------------------------

    def _check_cluster_ids(self):
        if self._cluster_ids is None:
            raise RuntimeError("The list of clusters need to be set.")

    def _filter(self, items):
        """Filter out ignored clusters or pairs of clusters."""
        return [item for item in items
                if item not in self._ignored]

    # Setting methods
    #--------------------------------------------------------------------------

    @property
    def cluster_ids(self):
        """Array of cluster ids in the current clustering."""
        return self._cluster_ids

    @cluster_ids.setter
    def cluster_ids(self, cluster_ids):
        """Update the array of cluster ids."""
        if isinstance(cluster_ids, np.ndarray):
            cluster_ids = cluster_ids.tolist()
        self._cluster_ids = sorted(cluster_ids)

    def set_similarity_function(self, func):
        """Register a function returing the similarity between two clusters."""
        self._similarity = func
        return func

    def set_quality_function(self, func):
        """Register a function returing the quality of a cluster."""
        self._quality = func
        return func

    # Core methods
    #--------------------------------------------------------------------------

    def best_clusters(self, n_max=None):
        """Return the list of best clusters sorted by decreasing quality.

        The registered quality function is used for the cluster quality.

        """
        self._check_cluster_ids()
        return self._filter(_best_clusters(self._cluster_ids, self._quality,
                                           n_max=n_max))

    def best_cluster(self):
        """Return the best cluster according to the registered cluster
        quality function."""
        clusters = self.best_clusters(n_max=1)
        if clusters:
            return clusters[0]

    def most_similar_clusters(self, cluster=None, n_max=None):
        """Return the `n_max` most similar clusters to a given cluster
        (the current best cluster by default)."""
        if cluster is None:
            cluster = self.best_cluster()
        self._check_cluster_ids()
        similarity = [(other, self._similarity(cluster, other))
                      for other in self._cluster_ids
                      if other != cluster]
        clusters = _argsort(similarity, n_max=n_max)
        # Filter out ignored clusters.
        clusters = self._filter(clusters)
        pairs = zip([cluster] * len(clusters), clusters)
        # Filter out ignored pairs of clusters.
        pairs = self._filter(pairs)
        return [clu for (_, clu) in pairs]

    def ignore(self, cluster_or_pair):
        """Mark a cluster or a pair of clusters as ignored.

        This cluster or pair of clusters will not reappear in the list of
        best clusters or most similar clusters.

        """
        if not isinstance(cluster_or_pair, (integer_types, tuple)):
            raise ValueError("This function accepts a cluster id "
                             "or a pair of ids as argument.")
        if isinstance(cluster_or_pair, tuple):
            assert len(cluster_or_pair) == 2
        self._ignored.add(cluster_or_pair)

    # List methods
    #--------------------------------------------------------------------------

    def next(self):
        if not self.is_running:
            self.start()
        # TODO

    def previous(self):
        pass

    def first(self):
        pass

    def last(self):
        pass

    # Playback methods
    #--------------------------------------------------------------------------

    def start(self):
        pass

    def pause(self):
        pass

    def stop(self):
        pass

    @property
    def is_running(self):
        pass

    # Pin methods
    #--------------------------------------------------------------------------

    def pin(self):
        pass

    def unpin(self):
        pass
