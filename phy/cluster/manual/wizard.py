# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from operator import itemgetter

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


def _find_first(items, filter=None):
    if not items:
        return None
    if filter is None:
        return items[0]
    return next(item for item in items if filter(item))


def _previous(items, current, filter=None):
    if current not in items:
        logger.debug("%d is not in %s.", current, items)
        return
    i = items.index(current)
    if i == 0:
        return current
    try:
        return _find_first(items[:i][::-1], filter)
    except StopIteration:
        return current


def _next(items, current, filter=None):
    if not items:
        return current
    if current not in items:
        logger.debug("%d is not in %s.", current, items)
        return
    i = items.index(current)
    if i == len(items) - 1:
        return current
    try:
        return _find_first(items[i + 1:], filter)
    except StopIteration:
        return current


def _progress(value, maximum):
    if maximum <= 1:
        return 1
    return int(100 * value / float(maximum - 1))


#------------------------------------------------------------------------------
# Wizard
#------------------------------------------------------------------------------

class Wizard(EventEmitter):
    """Propose a selection of high-quality clusters and merge candidates."""
    def __init__(self):
        super(Wizard, self).__init__()
        self._similarity = None
        self._quality = None
        self._get_cluster_ids = None
        self._cluster_status = lambda cluster: None
        self.reset()

    def reset(self):
        self._selection = []
        self._best_list = []  # This list is fixed (modulo clustering actions).
        self._match_list = []  # This list may often change.
        self._best = None
        self._match = None

    @property
    def has_started(self):
        return len(self._best_list) > 0

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

    # Internal methods
    #--------------------------------------------------------------------------

    def _with_status(self, items, status):
        """Filter out ignored clusters or pairs of clusters."""
        if not isinstance(status, (list, tuple)):
            status = [status]
        return [item for item in items if self._cluster_status(item) in status]

    def _is_not_ignored(self, cluster):
        return self._with_status([cluster], (None, 'good'))

    def _check(self):
        clusters = set(self.cluster_ids)
        assert set(self._best_list) <= clusters
        assert set(self._match_list) <= clusters
        if self._best is not None and len(self._best_list) >= 1:
            assert self._best in self._best_list
        if self._match is not None and len(self._match_list) >= 1:
            assert self._match in self._match_list
        if None not in (self.best, self.match):
            assert self.best != self.match

    def _sort(self, items, mix_good_unsorted=False):
        """Sort clusters according to their status:
        unsorted, good, and ignored."""
        if mix_good_unsorted:
            return (self._with_status(items, (None, 'good')) +
                    self._with_status(items, 'ignored'))
        else:
            return (self._with_status(items, None) +
                    self._with_status(items, 'good') +
                    self._with_status(items, 'ignored'))

    # Properties
    #--------------------------------------------------------------------------

    @property
    def cluster_ids(self):
        """Array of cluster ids in the current clustering."""
        return sorted(self._get_cluster_ids())

    # Core methods
    #--------------------------------------------------------------------------

    def cluster_status(self, cluster):
        return self._cluster_status(cluster)

    def best_clusters(self, n_max=None, quality=None):
        """Return the list of best clusters sorted by decreasing quality.

        The default quality function is the registered one.

        """
        if quality is None:
            quality = self._quality
        best = _best_clusters(self.cluster_ids, quality, n_max=n_max)
        return self._sort(best)

    def most_similar_clusters(self, cluster=None, n_max=None, similarity=None):
        """Return the `n_max` most similar clusters to a given cluster.

        The default similarity function is the registered one.

        """
        if cluster is None:
            cluster = self.best
            if cluster is None:
                cluster = self.best_clusters(1)[0]
        if similarity is None:
            similarity = self._similarity
        s = [(other, similarity(cluster, other))
             for other in self.cluster_ids
             if other != cluster]
        clusters = _argsort(s, n_max=n_max)
        return self._sort(clusters, mix_good_unsorted=True)

    # List methods
    #--------------------------------------------------------------------------

    def _set_best_list(self, cluster=None, clusters=None):
        if cluster is None:
            cluster = self.best
        if clusters is None:
            clusters = self.best_clusters()
        self._best_list = clusters
        if clusters:
            self.best = clusters[0]

    def _set_match_list(self, cluster=None, clusters=None):
        if cluster is None:
            cluster = self.best
        if clusters is None:
            clusters = self.most_similar_clusters(cluster)
        self._match_list = clusters
        if clusters:
            self.match = clusters[0]

    @property
    def best(self):
        """Currently-selected best cluster."""
        return self._best

    @best.setter
    def best(self, value):
        assert value in self._best_list
        self.selection = [value]

    @property
    def match(self):
        """Currently-selected closest match."""
        return self._match

    @match.setter
    def match(self, value):
        if value is not None:
            assert value in self._match_list
        if len(self._selection) == 1:
            self.selection = self.selection + [value]
        elif len(self._selection) == 2:
            self.selection = [self.selection[0], value]

    @property
    def selection(self):
        """Return the current best/match cluster selection."""
        return self._selection

    @selection.setter
    def selection(self, value):
        """Return the current best/match cluster selection."""
        assert isinstance(value, (tuple, list))
        clusters = self.cluster_ids
        value = [cluster for cluster in value if cluster in clusters]
        self._selection = value
        if len(self._selection) == 1:
            self._match = None
        if len(self._selection) >= 1:
            self._best = self._selection[0]
        if len(self._selection) >= 2:
            self._match = self._selection[1]
        self.emit('select', self._selection)

    @property
    def best_list(self):
        """Current list of best clusters, by decreasing quality."""
        return self._best_list

    @property
    def match_list(self):
        """Current list of closest matches, by decreasing similarity."""
        return self._match_list

    @property
    def n_processed(self):
        """Numbered of processed clusters so far.

        A cluster is considered processed if its status is not `None`.

        """
        return len(self._with_status(self._best_list, ('good', 'ignored')))

    @property
    def n_clusters(self):
        """Total number of clusters."""
        return len(self.cluster_ids)

    # Navigation
    #--------------------------------------------------------------------------

    def next_best(self):
        """Select the next best cluster."""
        self.best = _next(self._best_list,
                          self._best,
                          )
        if self.match is not None:
            self._set_match_list()

    def previous_best(self):
        """Select the previous best in cluster."""
        if self._best_list:
            self.best = _previous(self._best_list,
                                  self._best,
                                  )
        if self.match is not None:
            self._set_match_list()

    def next_match(self):
        """Select the next match."""
        # Handle the case where we arrive at the end of the match list.
        if self.match is not None and len(self._match_list) <= 1:
            self.next_best()
        elif self._match_list:
            self.match = _next(self._match_list,
                               self._match,
                               )

    def previous_match(self):
        """Select the previous match."""
        if self._match_list:
            self.match = _previous(self._match_list,
                                   self._match,
                                   )

    def next(self):
        """Next cluster proposition."""
        if self.match is None:
            return self.next_best()
        else:
            return self.next_match()

    def previous(self):
        """Previous cluster proposition."""
        if self.match is None:
            return self.previous_best()
        else:
            return self.previous_match()

    def first(self):
        """First match or first best."""
        if self.match is None:
            self.best = self._best_list[0]
        else:
            self.match = self._match_list[0]

    def last(self):
        """Last match or last best."""
        if self.match is None:
            self.best = self._best_list[-1]
        else:
            self.match = self._match_list[-1]

    # Control
    #--------------------------------------------------------------------------

    def start(self):
        """Start the wizard by setting the list of best clusters."""
        self._set_best_list()

    def pin(self, cluster=None):
        """Pin the current best cluster and set the list of closest matches."""
        if cluster is None:
            cluster = self.best
        logger.debug("Pin %d.", cluster)
        self.best = cluster
        self._set_match_list(cluster)
        self._check()

    def unpin(self):
        """Unpin the current cluster."""
        if self.match is not None:
            logger.debug("Unpin.")
            self.match = None
            self._match_list = []

    # Actions
    #--------------------------------------------------------------------------

    def _delete(self, clusters):
        for clu in clusters:
            if clu in self._best_list:
                self._best_list.remove(clu)
            if clu in self._match_list:
                self._match_list.remove(clu)
            if clu == self._best:
                self._best = self._best_list[0] if self._best_list else None
            if clu == self._match:
                self._match = None

    def _add(self, clusters, position=None):
        for clu in clusters:
            assert clu not in self._best_list
            assert clu not in self._match_list
            if self.best is not None:
                if position is not None:
                    self._best_list.insert(position, clu)
                else:  # pragma: no cover
                    self._best_list.append(clu)
            if self.match is not None:
                self._match_list.append(clu)

    def _update_state(self, up):
        # Update the cluster status.
        if up.description == 'metadata_group':
            cluster = up.metadata_changed[0]
            # Reorder the best list, so that the clusters moved in different
            # status go to their right place in the best list.
            if (self._best is not None and self._best_list and
                    cluster == self._best):
                # # Find the next best after the cluster has been moved.
                # next_best = _next(self._best_list, self._best)
                # Reorder the list.
                self._best_list = self._sort(self._best_list)
                # # Select the next best.
                # self._best = next_best
        # Update the wizard with new and old clusters.
        for clu in up.added:
            # Add the child at the parent's position.
            parents = [x for (x, y) in up.descendants if y == clu]
            parent = parents[0]
            position = (self._best_list.index(parent)
                        if self._best_list else None)
            self._add([clu], position)
        # Delete old clusters.
        self._delete(up.deleted)
        # # Select the last added cluster.
        # if self.best is not None and up.added:
        #     self._best = up.added[-1]

    def _select_after_update(self, up):
        if up.history == 'undo':
            self.selection = up.undo_state[0]['selection']
            return
        # Make as few updates as possible in the views after clustering
        # actions. This allows for better before/after comparisons.
        if up.added:
            self.selection = up.added
        if up.description == 'merge':
            self.pin(up.added[0])
        if up.description == 'metadata_group':
            cluster = up.metadata_changed[0]
            if cluster == self.best:
                # Pin the next best if there was a match before.
                match_before = self.match is not None
                self.next_best()
                if match_before:
                    self.pin()
            elif cluster == self.match:
                self.next_match()

    def attach(self, clustering, cluster_metadata):
        # TODO: might be better in an independent function in another module

        # The wizard gets the cluster ids from the Clustering instance
        # and the status from ClusterMetadataUpdater.
        self.set_cluster_ids_function(lambda: clustering.cluster_ids)

        @self.set_status_function
        def status(cluster):
            group = cluster_metadata.group(cluster)
            if group is None:  # pragma: no cover
                return None
            if group <= 1:
                return 'ignored'
            elif group == 2:
                return 'good'

        def on_request_undo_state(up):
            return {'selection': self.selection}

        def on_cluster(up):
            # Set the cluster metadata of new clusters.
            if up.added:
                cluster_metadata.set_from_descendants(up.descendants)
            # Update the wizard state.
            if self._best_list or self._match_list:
                self._update_state(up)
            # Make a new selection.
            if self._best is not None or self._match is not None:
                self._select_after_update(up)

        clustering.connect(on_request_undo_state)
        cluster_metadata.connect(on_request_undo_state)

        clustering.connect(on_cluster)
        cluster_metadata.connect(on_cluster)
