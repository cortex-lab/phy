# -*- coding: utf-8 -*-

"""Wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from operator import itemgetter


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

    def _remove_old_clusters(self, l):
        if not l:
            return l
        return [clu for clu in l if clu in self._cluster_ids]

    def _update_cluster_lists(self):
        self._list = self._remove_old_clusters(self._list)
        self._prev_list = self._remove_old_clusters(self._prev_list)
        if self._index is not None and self._index not in self._list:
            self._index = 0
        if self._prev_index is not None and self._prev_index not in self._list:
            self._prev_index = 0
        if self._pinned is not None and self._pinned not in self._list:
            self._pinned = None

    @cluster_ids.setter
    def cluster_ids(self, cluster_ids):
        """Update the list of clusters."""
        assert cluster_ids is not None
        self._cluster_ids = sorted(cluster_ids)
        self._update_cluster_lists()

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
        if isinstance(cluster_or_pair, tuple):
            assert len(cluster_or_pair) == 2
        else:
            cluster_or_pair = int(cluster_or_pair)
        self._ignored.add(cluster_or_pair)

    # List methods
    #--------------------------------------------------------------------------

    def _reset_list(self):

        # Current list.
        self._list = []
        self._index = None

        # Previous list (backup when pinning).
        self._prev_list = []
        self._prev_index = None

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

    def restart(self):
        self.stop()
        self.start()

    def is_running(self):
        return self._is_running

    def next(self):
        if not self._is_running:
            self.start()
        # Move to the next non-ignored.
        current = self._current
        if self._index is None:
            self._index = 0
        while self._current in self._ignored.union([current]):
            if self._index <= self.count() - 2:
                self._index += 1
            else:
                break
        return self._current

    def previous(self):
        current = self._current
        while self._current in self._ignored.union([current]):
            if self._is_running and 1 <= self._index:
                self._index -= 1
            else:
                break
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

    def pin(self, cluster=None):
        # Save the current list.
        self._prev_index = self._index
        self._prev_list = self._list
        if cluster is None:
            cluster = self._current
        if cluster is None:
            return
        assert cluster in self._cluster_ids
        self._pinned = cluster
        if self._pinned:
            self._list = self.most_similar_clusters(self._pinned)
            self._index = 0
        return self._pinned

    def unpin(self):
        self._pinned = None
        self._list = self._prev_list
        self._index = self._prev_index

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

    def set_best_clusters(self, clusters=None):
        if clusters is None:
            clusters = self.best_clusters()
        self._list = clusters
        self._index = 0


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
