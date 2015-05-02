# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from operator import itemgetter

from ...ext.six import integer_types
from ...utils.logging import debug


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _argsort(seq, reverse=True, n_max=None):
    """Return the list of clusters in decreasing order of value from
    a list of tuples (cluster, value)."""
    out = [cl for (cl, v) in sorted(seq,
                                    key=itemgetter(1),
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
    def __init__(self, cluster_ids):
        self._similarity = None
        self._quality = None
        self._ignored = set()
        self._reset_list()
        self.cluster_ids = cluster_ids

    # Internal methods
    #--------------------------------------------------------------------------

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
        """Update the list of clusters."""
        assert cluster_ids is not None
        self._cluster_ids = sorted(cluster_ids)
        if self._list:
            l = self._list
            self._list = [clu for clu in self._list
                          if clu in self._cluster_ids]
            changed = len(l) != len(self._list)
        else:
            changed = False
        if changed and self._index is not None and self._index > 0:
            debug("Reset the wizard because the list of clusters has changed.")

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

    def _reset_list(self):
        self._list = []
        self._index = None
        self._is_running = False
        self._pinned = None

    def count(self):
        return len(self._list)

    def index(self):
        return self._index

    def start(self):
        self._is_running = True
        if self._index is None:
            self.set_best_clusters()

    def pause(self):
        self._is_running = False

    def stop(self):
        self._reset_list()

    def is_running(self):
        return self._is_running

    def next(self):
        if not self._is_running:
            self.start()
        if self._index <= self.count() - 2:
            self._index += 1
        return self._current

    def previous(self):
        if self._is_running and self._index >= 1:
            self._index -= 1
        return self._current

    def first(self):
        self._index = 0
        return self._current

    def last(self):
        self._index = self.count() - 1
        return self._current

    @property
    def _current(self):
        if self._index is not None and 0 <= self._index < self.count():
            return self._list[self._index]

    # Pin methods
    #--------------------------------------------------------------------------

    def pin(self):
        self._pinned = self._current
        if self._pinned:
            self._list = self.most_similar_clusters(self._pinned)
            self._index = 0
        return self._pinned

    def unpin(self):
        self._pinned = None
        self._list = self.best_clusters()
        self._index = 0

    def pinned(self):
        return self._pinned

    def current_selection(self):
        if not self._is_running:
            return ()
        current = self._current
        assert current is not None
        # Best unsorted.
        if self._pinned is None:
            return (current,)
        # Best unsorted and closest match.
        else:
            return (self._pinned, current)

    def ignore_current_selection(self):
        self.ignore(self.current_selection())
        self.next()

    def set_best_clusters(self):
        self._index = 0
        self._list = self.best_clusters()
