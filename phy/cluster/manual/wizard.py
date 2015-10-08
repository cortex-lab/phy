# -*- coding: utf-8 -*-
"""Wizard."""

#------------------------------------------------------------------------------
# Imports

#------------------------------------------------------------------------------

from itertools import product
import logging
from operator import itemgetter

from six import string_types

from ._history import History
from phy.utils import EventEmitter

logger = logging.getLogger(__name__)


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


def _next_in_list(l, item):
    if l and item in l and l.index(item) < len(l) - 1:
        return l[l.index(item) + 1]
    return item


def _sort_by_status(clusters, status=None, remove_ignored=False):
    """Sort clusters according to their status."""
    assert status
    _sort_map = {None: 0, 'good': 1, 'ignored': 2}
    if remove_ignored:
        clusters = [c for c in clusters if status(c) != 'ignored']
    # NOTE: sorted is "stable": it doesn't change the order of elements
    # that compare equal, which ensures that the order of clusters is kept
    # among any given status.
    key = lambda cluster: _sort_map.get(status(cluster), 0)
    return sorted(clusters, key=key)


def _best_clusters(clusters, quality, n_max=None):
    return _argsort([(cluster, quality(cluster))
                     for cluster in clusters], n_max=n_max)


def _most_similar_clusters(cluster, cluster_ids=None, n_max=None,
                           similarity=None, status=None, less_than=None):
    """Return the `n_max` most similar clusters to a given cluster."""
    if cluster not in cluster_ids:
        return []
    s = [(other, similarity(cluster, other))
         for other in cluster_ids
         if other != cluster and status(other) != 'ignored']
    # Only keep values less than a threshold.
    if less_than:
        s = [(c, v) for (c, v) in s if v <= less_than]
    clusters = _argsort(s, n_max=n_max)
    return _sort_by_status(clusters, status=status, remove_ignored=True)


def _wizard_group(group):
    # The group should be None, 'mua', 'noise', or 'good'.
    assert group is None or isinstance(group, string_types)
    group = group.lower() if group else group
    if group in ('mua', 'noise'):
        return 'ignored'
    elif group == 'good':
        return 'good'
    return None


#------------------------------------------------------------------------------
# Strategy functions
#------------------------------------------------------------------------------

def _best_quality_strategy(selection,
                           cluster_ids=None,
                           quality=None,
                           status=None,
                           similarity=None):
    """Two cases depending on the number of selected clusters:

    * 1: move to the next best cluster
    * 2: move to the next most similar pair
    * 3+: do nothing

    """
    if selection is None:
        return selection
    selection = tuple(selection)
    n = len(selection)
    if n <= 1:
        best_clusters = _best_clusters(cluster_ids, quality)
        # Sort the best clusters according to their status.
        best_clusters = _sort_by_status(best_clusters, status=status)
        if selection:
            return (_next_in_list(best_clusters, selection[0]),)
        elif best_clusters:
            return (best_clusters[0],)
        else:  # pragma: no cover
            return selection
    elif n == 2:
        best, match = selection
        value = similarity(best, match)
        candidates = _most_similar_clusters(best,
                                            cluster_ids=cluster_ids,
                                            similarity=similarity,
                                            status=status,
                                            less_than=value)
        if best in candidates:  # pragma: no cover
            candidates.remove(best)
        if match in candidates:
            candidates.remove(match)
        if not candidates:
            return selection
        return (best, candidates[0])


def _best_similarity_strategy(selection,
                              cluster_ids=None,
                              quality=None,
                              status=None,
                              similarity=None):
    if selection is None:
        return selection
    selection = tuple(selection)
    n = len(selection)
    if n >= 2:
        best, match = selection
        value = similarity(best, match)
    else:
        best, match = None, None
        value = None
    # We remove the current pair, the (x, x) pairs, and we ensure that
    # (d, c) doesn't appear if (c, d) does. We choose the pair where
    # the first cluster of the pair has the highest quality.
    # Finally we remove the ignored clusters.
    s = [((c, d), similarity(c, d))
         for c, d in product(cluster_ids, repeat=2)
         if c != d and (c, d) != (best, match)
         and quality(c) >= quality(d)
         and status(c) != 'ignored'
         and status(d) != 'ignored'
         ]

    if value is not None:
        s = [((c, d), v) for ((c, d), v) in s if v <= value]
    pairs = _argsort(s)
    if pairs:
        return pairs[0]
    else:
        return selection


#------------------------------------------------------------------------------
# Wizard
#------------------------------------------------------------------------------

class Wizard(EventEmitter):
    """Propose a selection of high-quality clusters and merge candidates.

    * The wizard is responsible for the selected clusters.
    * The wizard keeps no state about the clusters: the state is entirely
      provided by functions: cluster_ids, status (group), similarity, quality.
    * The wizard keeps track of the history of the selected clusters, but this
      history is cleared after every action that changes the state.
    * The `next()` function proposes a new selection as a function of the
      current selection only.
    * There are two strategies: best-quality or best-similarity strategy.

    TODO: cache expensive functions.

    """
    def __init__(self):
        super(Wizard, self).__init__()
        self._similarity = None
        self._quality = None
        self._get_cluster_ids = None
        self._cluster_status = lambda cluster: None
        self._selection = ()
        self.reset()

    def reset(self):
        self._history = History(())

    # Quality and status functions
    #--------------------------------------------------------------------------

    def set_cluster_ids_function(self, func):
        """Register a function giving the list of cluster ids."""
        self._get_cluster_ids = func

    def set_status_function(self, func):
        """Register a function returning the status of a cluster: None,
        'ignored', or 'good'.

        Can be used as a decorator.

        """
        self._cluster_status = func
        return func

    def set_similarity_function(self, func):
        """Register a function returning the similarity between two clusters.

        Can be used as a decorator.

        """
        self._similarity = func
        return func

    def set_quality_function(self, func):
        """Register a function returning the quality of a cluster.

        Can be used as a decorator.

        """
        self._quality = func
        return func

    # Properties
    #--------------------------------------------------------------------------

    @property
    def cluster_ids(self):
        """Array of cluster ids in the current clustering."""
        return sorted(self._get_cluster_ids())

    @property
    def n_clusters(self):
        """Total number of clusters."""
        return len(self.cluster_ids)

    # Selection methods
    #--------------------------------------------------------------------------

    def select(self, cluster_ids, add_to_history=True):
        if cluster_ids is None:  # pragma: no cover
            return
        assert hasattr(cluster_ids, '__len__')
        clusters = self.cluster_ids
        cluster_ids = tuple(cluster for cluster in cluster_ids
                            if cluster in clusters)
        self._selection = cluster_ids
        if add_to_history:
            self._history.add(self._selection)
        self.emit('select', self._selection)

    @property
    def selection(self):
        """Return the current cluster selection."""
        return self._selection

    @selection.setter
    def selection(self, value):
        self.select(value)

    @property
    def best(self):
        """Currently-selected best cluster."""
        return self._selection[0] if self._selection else None

    @property
    def match(self):
        """Currently-selected closest match."""
        return self._selection[1] if len(self._selection) >= 2 else None

    def pin(self):
        """Select the cluster the most similar cluster to the current best."""
        best = self.best
        if best is None:
            return
        candidates = _most_similar_clusters(best,
                                            cluster_ids=self.cluster_ids,
                                            similarity=self._similarity,
                                            status=self._cluster_status)
        assert best not in candidates
        if not candidates:
            return
        self.select((self.best, candidates[0]))

    def unpin(self):
        if len(self._selection) == 2:
            self.selection = (self.selection[0],)

    # Navigation
    #--------------------------------------------------------------------------

    def _set_selection_from_history(self):
        cluster_ids = self._history.current_item
        if not cluster_ids:  # pragma: no cover
            return
        self.select(cluster_ids, add_to_history=False)

    def previous(self):
        if self._history.current_position <= 2:
            return self._selection
        self._history.back()
        self._set_selection_from_history()
        return self._selection

    def next(self):
        if not self._history.is_last():
            # Go forward after a previous.
            self._history.forward()
            self._set_selection_from_history()

    def next_by_quality(self):
        self.selection = _best_quality_strategy(
            self._selection,
            cluster_ids=self._get_cluster_ids(),
            quality=self._quality,
            status=self._cluster_status,
            similarity=self._similarity)
        return self._selection

    def next_by_similarity(self):
        self.selection = _best_similarity_strategy(
            self._selection,
            cluster_ids=self._get_cluster_ids(),
            quality=self._quality,
            status=self._cluster_status,
            similarity=self._similarity)
        return self._selection
