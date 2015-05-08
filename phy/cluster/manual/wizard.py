# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from operator import itemgetter

from ...utils.array import _is_array_like
from ...utils.logging import info


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
        raise RuntimeError("{0} is not in {1}.".format(current, items))
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
        raise RuntimeError("{0} is not in {1}.".format(current, items))
    i = items.index(current)
    if i == len(items) - 1:
        return current
    try:
        return _find_first(items[i + 1:], filter)
    except StopIteration:
        return current


#------------------------------------------------------------------------------
# Wizard
#------------------------------------------------------------------------------

class Wizard(object):
    """Propose a selection of high-quality clusters and merge candidates."""
    def __init__(self, cluster_groups):
        self._similarity = None
        self._quality = None
        # cluster_groups is a dictionary or is converted to one.
        if _is_array_like(cluster_groups):
            # A group can be None (unsorted), `good`, or `ignored`.
            cluster_groups = {clu: None for clu in cluster_groups}
        self._cluster_groups = cluster_groups
        self._best_list = []  # This list is fixed (modulo clustering actions).
        self._match_list = []  # This list may often change.
        self._best = None
        self._match = None

    # Quality functions
    #--------------------------------------------------------------------------

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

    def _group(self, cluster):
        return self._cluster_groups.get(cluster, None)

    def _in_groups(self, items, groups):
        """Filter out ignored clusters or pairs of clusters."""
        if not isinstance(groups, (list, tuple)):
            groups = [groups]
        return [item for item in items if self._group(item) in groups]

    def _is_not_ignored(self, cluster):
        return self._in_groups([cluster], (None, 'good'))

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
        """Sort clusters according to their groups:
        unsorted, good, and ignored."""
        if mix_good_unsorted:
            return (self._in_groups(items, (None, 'good')) +
                    self._in_groups(items, 'ignored'))
        else:
            return (self._in_groups(items, None) +
                    self._in_groups(items, 'good') +
                    self._in_groups(items, 'ignored'))

    # Properties
    #--------------------------------------------------------------------------

    @property
    def cluster_ids(self):
        """Array of cluster ids in the current clustering."""
        return sorted(self._cluster_groups)

    @property
    def cluster_groups(self):
        """Dictionary with the groups of each cluster.

        The groups are: `None` (corresponds to unsorted), `good`, or `ignored`.

        """
        return self._cluster_groups

    # Core methods
    #--------------------------------------------------------------------------

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
        self._best = value

    @property
    def match(self):
        """Currently-selected closest match."""
        return self._match

    @match.setter
    def match(self, value):
        if value is not None:
            assert value in self._match_list
        self._match = value

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

        A cluster is considered processed if its group is not `None`.

        """
        return len(self._in_groups(self._best_list, ('good', 'ignored')))

    @property
    def n_clusters(self):
        """Total number of clusters."""
        return len(self.cluster_ids)

    # Navigation
    #--------------------------------------------------------------------------

    def next_best(self):
        """Select the next best cluster."""
        # Handle the case where we arrive at the end of the best list.
        if self.best is not None and len(self._best_list) <= 1:
            info("The wizard has finished.")
        else:
            self.best = _next(self._best_list,
                              self._best,
                              # self._is_not_ignored,
                              )
        if self.match is not None:
            self._set_match_list()

    def previous_best(self):
        """Select the previous best in cluster."""
        self.best = _previous(self._best_list,
                              self._best,
                              # self._is_not_ignored,
                              )
        if self.match is not None:
            self._set_match_list()

    def next_match(self):
        """Select the next match."""
        # Handle the case where we arrive at the end of the match list.
        if self.match is not None and len(self._match_list) <= 1:
            self.next_best()
        else:
            self.match = _next(self._match_list,
                               self._match,
                               # self._is_not_ignored,
                               )

    def previous_match(self):
        """Select the previous match."""
        self.match = _previous(self._match_list,
                               self._match,
                               # self._is_not_ignored,
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
        if self.match is not None and self.best == cluster:
            return
        self.best = cluster
        self._set_match_list(cluster)

    def unpin(self):
        """Unpin the current cluster."""
        if self.match is not None:
            self.match = None
            self._match_list = []

    # Actions
    #--------------------------------------------------------------------------

    def _delete(self, clusters):
        for clu in clusters:
            if clu in self._cluster_groups:
                del self._cluster_groups[clu]
            if clu in self._best_list:
                self._best_list.remove(clu)
            if clu in self._match_list:
                self._match_list.remove(clu)

    def _add(self, clusters, group):
        for clu in clusters:
            self._cluster_groups[clu] = group
            if self.best is not None:
                index = self._best_list.index(self.best)
                self._best_list.insert(index, clu)

    def move(self, cluster, group):
        """Move a cluster to a group."""
        self._cluster_groups[cluster] = group
        if cluster == self.best:
            self.next_best()
        elif cluster == self.match:
            self.next_match()
        self._check()

    def merge(self, old, new, group):
        """Merge a list of clusters to a new cluster."""
        if isinstance(new, (tuple, list)):
            assert len(new) == 1
            new = new[0]
        # Add new cluster.
        self._add([new], group)
        # Delete old clusters.
        self._delete(old)
        # Pin the newly-created cluster.
        if self.best in old and self.match is not None and self.match in old:
            self.pin(new)
        # TODO: the match list is not updated currently; it could be if
        # necessary.
        self._check()

    def update_clusters(self, deleted, added, group):
        """Update the list of clusters.

        Specify the lists of deleted and added clusters.

        """
        self._delete(deleted)
        self._add(added, group)
        # Update the best cluster if it was deleted.
        if self.best in deleted and self.match is not None:
            self.pin(added[0])
        self._check()

    # Panel
    #--------------------------------------------------------------------------

    @property
    def _best_progress(self):
        """Progress in the best clusters."""
        value = self.best_list.index(self.best)
        maximum = len(self.best_list)
        return _progress(value, maximum)

    @property
    def _match_progress(self):
        """Progress in the processed clusters."""
        value = self.n_processed
        maximum = self.n_clusters
        return _progress(value, maximum)

    def _repr_html_(self):
        return _wizard_panel_html(best=self.best,
                                  match=self.match,
                                  best_progress=self._best_progress,
                                  match_progress=self._match_progress,
                                  best_group=self._group(self.best),
                                  match_group=self._group(self.match),
                                  )


#------------------------------------------------------------------------------
# Wizard panel
#------------------------------------------------------------------------------

_PANEL_HTML = """
<div class="control-panel">
    <div class="best">
        <div class="id {best_group}">{best}</div>
        <div class="progress">
            <progress value="{best_progress:d}" max="100"></progress>
        </div>
    </div>
    <div class="match">
        <div class="id {match_group}">{match}</div>
        <div class="progress">
            <progress value="{match_progress:d}" max="100"></progress>
        </div>
    </div>
</div>"""


_PANEL_CSS = """
html, body, div {
    background-color: black;
}

.control-panel {
    background-color: black;
    color: white;
    font-weight: bold;
    font-size: 24pt;
    padding: 10px;
    text-align: center
}

.control-panel > div {
    display: inline-block;
    margin: 0 auto;
}

.control-panel .best {
    margin-right: 20px;
    color: rgb(102, 194, 165);
}

.control-panel .match {
    color: rgb(252, 141, 98);
}

.control-panel > div .id {
    margin: 10px 0 20px 0;
    height: 40px;
    text-align: center;
    vertical-align: middle;
    padding: 5px 0 10px 0;
}

/* Progress bar */
.control-panel progress[value] {
    width: 200px;
}

/* Cluster group */
.control-panel .unsorted {
    /*background-color: #333;*/
}

.control-panel .good {
    background-color: #243318;
}

.control-panel .ignored {
    background-color: #331d12;
}

"""


def _wizard_panel_html(best=None,
                       match=None,
                       best_progress=None,
                       match_progress=None,
                       best_group=None,
                       match_group=None,
                       ):
    out = '<style>' + _PANEL_CSS + '</style>\n'
    out += _PANEL_HTML.format(best=best if best is not None else '',
                              match=match if match is not None else '',
                              best_progress=best_progress,
                              match_progress=match_progress,
                              best_group=best_group or 'unsorted',
                              match_group=match_group or 'unsorted',
                              )
    return out


def _progress(value, maximum):
    if maximum <= 1:
        return 1
    return int(100 * value / float(maximum - 1))
