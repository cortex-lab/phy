# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from operator import itemgetter

from ...utils.array import _is_array_like


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


def _find_first(items, filter):
    return next(item for item in items if filter(item))


def _previous(items, current, filter):
    if current not in items:
        raise RuntimeError("{0} is not in {1}.".format(current, items))
    i = items.index(current)
    if i == 0:
        return current
    try:
        return _find_first(items[:i][::-1], filter)
    except StopIteration:
        return current


def _next(items, current, filter):
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
            # A group can be None (unsorted), 'good', or 'ignored'.
            cluster_groups = {clu: None for clu in cluster_groups}
        self._cluster_groups = cluster_groups
        self._best_list = []  # This list is fixed (modulo clustering actions).
        self._match_list = []  # This list may often change.
        self._best = None
        self._match = None

    # Quality functions
    #--------------------------------------------------------------------------

    def set_similarity_function(self, func):
        """Register a function returing the similarity between two clusters."""
        self._similarity = func
        return func

    def set_quality_function(self, func):
        """Register a function returing the quality of a cluster."""
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

    def _check(self):
        clusters = set(self.cluster_ids)
        assert set(self._best_list) <= clusters
        assert set(self._match_list) <= clusters
        if self._best is not None:
            assert self._best in self._best_list
        if self._match is not None:
            assert self._match in self._match_list

    def _sort(self, items):
        """Sort clusters according to their groups:
        unsorted, good, and ignored."""
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
        return self._cluster_groups

    # Actions
    #--------------------------------------------------------------------------

    def move(self, cluster, group):
        self._groups[cluster] = group
        self._check()
        if cluster == self.best:
            self.next_best()
        elif cluster == self.match:
            self.next_match()

    def merge(self, old, new, group):
        b, m = self.best, self.match
        assert b is not None and m is not None
        # Add new cluster.
        self._groups[new] = group
        index = self._best_list.index(b)
        self._best_list.insert(index, new)
        # Delete old clusters.
        for clu in old:
            del self._groups[clu]
            if clu in self._best_list:
                self._best_list.remove(clu)
            if clu in self._match_list:
                self._match_list.remove(clu)
        self._check()
        # Update current selection.
        # if sorted(old) == sorted([b, m]):
        self.best = new
        self.set_match_list()

    # Core methods
    #--------------------------------------------------------------------------

    def best_clusters(self, n_max=None):
        """Return the list of best clusters sorted by decreasing quality.

        The registered quality function is used for the cluster quality.

        """
        best = _best_clusters(self.cluster_ids, self._quality, n_max=n_max)
        return self._sort(best)

    # def best_cluster(self):
    #     """Return the best cluster according to the registered cluster
    #     quality function."""
    #     clusters = self.best_clusters(n_max=1)
    #     if clusters:
    #         return clusters[0]

    def most_similar_clusters(self, cluster=None, n_max=None):
        """Return the `n_max` most similar clusters to a given cluster
        (the current best cluster by default)."""
        if cluster is None:
            cluster = self.best
            if cluster is None:
                cluster = self.best_clusters(1)[0]
        similarity = [(other, self._similarity(cluster, other))
                      for other in self.cluster_ids
                      if other != cluster]
        clusters = _argsort(similarity, n_max=n_max)
        return self._sort(clusters)

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
        return self._best

    @best.setter
    def best(self, value):
        assert value in self._best_list
        self._best = value

    @property
    def match(self):
        return self._match

    @match.setter
    def match(self, value):
        if value is None:
            return ValueError("The match needs to be a valid cluster id.")
        assert value in self._match_list
        self._match = value

    def next_best(self):
        self.best = _next(self._best_list, self._best)

    def previous_best(self):
        self.best = _previous(self._best_list, self._best)

    def next_match(self):
        self.match = _next(self._match_list, self._match)

    def previous_match(self):
        self.match = _previous(self._match_list, self._match)

    def next(self):
        if self.match is None:
            return self.next_best()
        else:
            return self.next_match()

    def previous(self):
        if self.match is None:
            return self.previous_best()
        else:
            return self.previous_match()

    def first(self):
        self.best = self._best_list[0]

    def last(self):
        self.best = self._best_list[-1]

    # Control
    #--------------------------------------------------------------------------

    def start(self):
        self._set_best_list()

    def pin(self, cluster=None):
        if cluster is None:
            cluster = self.best
        self._set_match_list(cluster)

    def unpin(self):
        if self.match is not None:
            self.match = None
            self._match_list = []


#------------------------------------------------------------------------------
# Wizard panel
#------------------------------------------------------------------------------

_PANEL_HTML = """
<div class="control-panel">
<div class="best">
    <div class="id">{best}</div>
    <div class="progress">
        <progress value="{best_progress:d}" max="100"></progress>
    </div>
</div>
<div class="match">
    <div class="id">{match}</div>
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
}

.control-panel progress[value] {
    width: 200px;
}

"""


def _wizard_panel_html(best=None,
                       best_progress=None,
                       match=None,
                       match_progress=None,
                       ):
    out = '<style>' + _PANEL_CSS + '</style>\n'
    out += _PANEL_HTML.format(best=best,
                              best_progress=best_progress,
                              match=match,
                              match_progress=match_progress,
                              )
    return out


class WizardPanel(object):
    def __init__(self):
        self._best = None
        self._match = None
        self._best_index = 0
        self._match_index = 0
        self._best_count = 0
        self._match_count = 0

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value):
        self._best = value

    @property
    def match(self):
        return self._match

    @match.setter
    def match(self, value):
        self._match = value

    @property
    def best_index(self):
        return self._best_index

    @best_index.setter
    def best_index(self, value):
        self._best_index = value

    @property
    def best_count(self):
        return self._best_count

    @best_count.setter
    def best_count(self, value):
        self._best_count = value

    @property
    def match_index(self):
        return self._match_index

    @match_index.setter
    def match_index(self, value):
        self._match_index = value

    @property
    def match_count(self):
        return self._match_count

    @match_count.setter
    def match_count(self, value):
        self._match_count = value

    def _progress(self, value, maximum):
        if maximum <= 1:
            return 1
        return int(100 * value / float(maximum - 1))

    @property
    def html(self):
        bp = self._progress(self.best_index, self.best_count)
        mp = self._progress(self.match_index, self.match_count)
        return _wizard_panel_html(best=self.best
                                  if self.best is not None else '',
                                  match=self.match
                                  if self.match is not None else '',
                                  best_progress=bp,
                                  match_progress=mp,
                                  )

    def _repr_html_(self):
        return self.html
