# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..wizard import (_argsort,
                      _sort,
                      _next_in_list,
                      _best_clusters,
                      _most_similar_clusters,
                      _wizard_group,
                      best_quality_strategy,
                      )
from .._utils import UpdateInfo
from phy.utils import EventEmitter


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_argsort():
    l = [(1, .1), (2, .2), (3, .3), (4, .4)]
    assert _argsort(l) == [4, 3, 2, 1]

    assert _argsort(l, n_max=0) == [4, 3, 2, 1]
    assert _argsort(l, n_max=1) == [4]
    assert _argsort(l, n_max=2) == [4, 3]
    assert _argsort(l, n_max=10) == [4, 3, 2, 1]

    assert _argsort(l, reverse=False) == [1, 2, 3, 4]


def test_sort():
    clusters = [10, 0, 1, 30, 2, 20]
    #            N, i, g,  N, N,  N
    status = lambda c: ('ignored', 'good')[c] if c <= 1 else None

    assert _sort(clusters, status=status) == [10, 30, 2, 20, 1, 0]
    assert _sort(clusters, status=status, remove_ignored=True) == \
        [10, 30, 2, 20, 1]


def test_best_clusters():
    quality = lambda c: c * .1
    l = list(range(1, 5))
    assert _best_clusters(l, quality) == [4, 3, 2, 1]
    assert _best_clusters(l, quality, n_max=0) == [4, 3, 2, 1]
    assert _best_clusters(l, quality, n_max=1) == [4]
    assert _best_clusters(l, quality, n_max=2) == [4, 3]
    assert _best_clusters(l, quality, n_max=10) == [4, 3, 2, 1]


def test_most_similar_clusters():
    cluster_ids = [0, 1, 2, 3]
    #              i, g, N, i
    similarity = lambda c, d: c + d
    status = lambda c: ('ignored', 'good', None, 'ignored')[c]

    def _similar(cluster):
        return _most_similar_clusters(cluster,
                                      cluster_ids=cluster_ids,
                                      similarity=similarity,
                                      status=status)

    assert not _similar(None)
    assert not _similar(10)
    assert _similar(0) == [2, 1]
    assert _similar(1) == [2]
    assert _similar(2) == [1]


def test_next_in_list():
    l = [1, 2, 3]
    assert _next_in_list(l, 0) == 0
    assert _next_in_list(l, 1) == 2
    assert _next_in_list(l, 2) == 3
    assert _next_in_list(l, 3) == 3
    assert _next_in_list(l, 4) == 4


def test_best_quality_strategy():
    cluster_ids = [0, 1, 2, 3, 4]
    #              i, i, g, N, N
    quality = lambda c: c
    status = lambda c: ('ignored', 'ignored', 'good')[c] if c <= 2 else None
    similarity = lambda c, d: c + d

    def _next(selection):
        return best_quality_strategy(selection,
                                     cluster_ids=cluster_ids,
                                     quality=quality,
                                     status=status,
                                     similarity=similarity)

    assert not _next(None)
    assert _next(()) == (4,)

    for i in range(4, -1, -1):
        assert _next((i,)) == (max(0, i - 1),)

    assert _next((4, 3)) == (4, 2)
    assert _next((4, 2)) == (4, 2)  # 1 is ignored, so it does not appear.

    assert _next((3, 2)) == (3, 2)
    assert _next((2, 3)) == (2, 3)


def test_wizard_group():
    assert _wizard_group('noise') == 'ignored'
    assert _wizard_group('mua') == 'ignored'
    assert _wizard_group('good') == 'good'
    assert _wizard_group('unknown') is None
    assert _wizard_group(None) is None


def test_wizard_nav(mock_wizard):
    w = mock_wizard
    assert w.cluster_ids == [1, 2, 3]
    assert w.n_clusters == 3

    assert w.selection == ()

    ###
    w.selection = []
    assert w.selection == ()

    assert w.best is None
    assert w.match is None

    ###
    w.selection = [1]
    assert w.selection == (1,)

    assert w.best == 1
    assert w.match is None

    ###
    w.select([1, 2, 4])
    assert w.selection == (1, 2)

    assert w.best == 1
    assert w.match == 2

    ###
    w.previous()
    assert w.selection == (1,)

    for _ in range(2):
        w.previous()
        assert w.selection == (1,)

    ###
    w.next()
    assert w.selection == (1, 2)

    for _ in range(2):
        w.next()
        assert w.selection == (1, 2)


def test_wizard_strategy_1(mock_wizard):
    w = mock_wizard

    w.set_status_function(lambda x: None)

    def strategy(selection, cluster_ids=None, quality=None,
                 status=None, similarity=None):
        """Return the next best cluster."""
        best_clusters = _best_clusters(cluster_ids, quality)
        if not selection:
            return (best_clusters[0],)
        assert len(selection) == 1
        return (_next_in_list(best_clusters, selection[0]),)

    w.set_strategy_function(strategy)
    assert w.selection == ()

    for i in range(3, 0, -1):
        w.next()
        assert w.selection == (i,)

    w.next()
    assert w.selection == (1,)


def test_wizard_strategy_groups(mock_wizard, cluster_groups):
    w = mock_wizard

    w.attach_cluster_groups(cluster_groups)

    assert 101 in w.cluster_ids
    assert 105 in w.cluster_ids

    w.pin()
    assert w.selection == ()

    for i in range(105, 100, -1):
        assert w.next() == (i,)

    w.select([105])

    w.pin()
    assert w.selection == (105, 104)

    w.next()
    assert w.selection == (105, 103)

    w.previous()
    assert w.selection == (105, 104)

    w.unpin()
    assert w.selection == (105,)

    @w.set_status_function
    def status(cluster):
        return 'ignored'

    assert w.pin() is None


def test_wizard_attach(mock_wizard):
    w = mock_wizard

    def strategy(selection, **kwargs):
        if not selection:
            return (3,)
        return (1 + (selection[0] % 3),)

    w.set_strategy_function(strategy)
    w.select([3])

    obj = EventEmitter()
    w.attach(obj)

    def _action(**kwargs):
        up = UpdateInfo(**kwargs)
        return obj.emit('cluster', up)

    _action(description='merge', added=[3])
    assert w.selection == (3,)

    _action(history='undo', undo_state=[{'selection': w.selection}])
    assert w.selection == (3,)

    _action(history='redo')
    assert w.selection == (1,)

    assert obj.emit('request_undo_state', {}) == [{'selection': (w.selection)}]

    w.select((1, 2))
    _action(description='metadata_group', metadata_changed=[1])
    assert w.selection == (1, 3)

    _action(description='metadata_group', metadata_changed=[3])
    assert w.selection == (2,)
