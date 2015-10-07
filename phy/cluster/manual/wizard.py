# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from operator import itemgetter

from six import string_types

from ._history import History
from phy.utils._types import _as_list, _as_tuple
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


def _best_clusters(clusters, quality, n_max=None):
    return _argsort([(cluster, quality(cluster))
                     for cluster in clusters], n_max=n_max)


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
        self._next = None  # Strategy function.
        self.reset()

    def reset(self):
        self._selection = ()
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

    def set_strategy_function(self, func):
        """Register a function returning a new selection after the current
        selection, as a function of the quality and similarity of the clusters.
        """
        # func(selection, cluster_ids=None, quality=None, similarity=None)

        def wrapped(sel):
            return func(self._selection,
                        cluster_ids=self._get_cluster_ids(),
                        quality=self._quality,
                        similarity=self._similarity,
                        )

        self._next = wrapped

    # Internal methods
    #--------------------------------------------------------------------------

    def _sort_nomix(self, cluster):
        # Sort by unsorted first, good second, ignored last.
        _sort_map = {None: 0, 'good': 1, 'ignored': 2}
        return _sort_map.get(self._cluster_status(cluster), 0)

    def _sort_mix(self, cluster):
        # Sort by unsorted/good first, ignored last.
        _sort_map = {None: 0, 'good': 0, 'ignored': 2}
        return _sort_map.get(self._cluster_status(cluster), 0)

    def _sort(self, clusters, mix_good_unsorted=False):
        """Sort clusters according to their status."""
        key = self._sort_mix if mix_good_unsorted else self._sort_nomix
        return sorted(clusters, key=key)

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

    # Core methods
    #--------------------------------------------------------------------------

    def cluster_status(self, cluster):
        return self._cluster_status(cluster)

    def best_clusters(self, n_max=None, quality=None):
        """Return the list of best clusters sorted by decreasing quality.

        The default quality function is the registered one.

        """
        quality = quality or self._quality
        best = _best_clusters(self.cluster_ids, quality, n_max=n_max)
        return self._sort(best)

    def most_similar_clusters(self, cluster, n_max=None, similarity=None):
        """Return the `n_max` most similar clusters to a given cluster.

        The default similarity function is the registered one.

        """
        similarity = similarity or self._similarity
        s = [(other, similarity(cluster, other))
             for other in self.cluster_ids
             if other != cluster]
        clusters = _argsort(s, n_max=n_max)
        return self._sort(clusters, mix_good_unsorted=True)

    # Selection methods
    #--------------------------------------------------------------------------

    @property
    def selection(self):
        """Return the current cluster selection."""
        return _as_tuple(self._selection)

    @selection.setter
    def selection(self, value):
        clusters = self.cluster_ids
        value = tuple(cluster for cluster in value if cluster in clusters)
        self._selection = value
        self.emit('select', self._selection)

    @property
    def best(self):
        """Currently-selected best cluster."""
        return self._selection[0] if self._selection else None

    @property
    def match(self):
        """Currently-selected closest match."""
        return self._selection[1] if len(self._selection) >= 2 else None

    # Navigation
    #--------------------------------------------------------------------------

    def previous(self):
        sel = self._history.back()
        if sel:
            self._selection = tuple(sel)

    def next(self):
        if not self._history.is_last():
            # Go forward after a previous.
            sel = self._history.forward()
            if sel:
                self._selection = tuple(sel)
        else:
            # Or compute the next selection.
            self._selection = tuple(self._next(self._selection))
            self._history.add(self._selection)

    # Attach
    #--------------------------------------------------------------------------

    def attach(self, obj):
        """Attach an actioner to the wizard."""

        # Save the current selection when an action occurs.
        @obj.connect
        def on_request_undo_state(up):
            return {'selection': self._selection}

        @obj.connect
        def on_cluster(up):
            if up.history == 'undo':
                # Revert to the given selection after an undo.
                self._selection = tuple(up.undo_state[0]['selection'])
            else:
                # Or move to the next selection after any other action.
                self.next()
