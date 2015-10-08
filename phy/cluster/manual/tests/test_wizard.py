# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..wizard import (_argsort,
                      _sort_by_status,
                      _next_in_list,
                      _best_clusters,
                      _most_similar_clusters,
                      _best_quality_strategy,
                      _best_similarity_strategy,
                      Wizard,
                      )


#------------------------------------------------------------------------------
# Test utility functions
#------------------------------------------------------------------------------

def test_argsort():
    l = [(1, .1), (2, .2), (3, .3), (4, .4)]
    assert _argsort(l) == [4, 3, 2, 1]

    assert _argsort(l, n_max=0) == [4, 3, 2, 1]
    assert _argsort(l, n_max=1) == [4]
    assert _argsort(l, n_max=2) == [4, 3]
    assert _argsort(l, n_max=10) == [4, 3, 2, 1]

    assert _argsort(l, reverse=False) == [1, 2, 3, 4]


def test_sort_by_status(status):
    cluster_ids = [10, 0, 1, 30, 2, 20]
    assert _sort_by_status(cluster_ids, status=status) == \
        [30, 2, 20, 1, 10, 0]
    assert _sort_by_status(cluster_ids, status=status,
                           remove_ignored=True) == [30, 2, 20, 1]


def test_next_in_list():
    l = [1, 2, 3]
    assert _next_in_list(l, 0) == 0
    assert _next_in_list(l, 1) == 2
    assert _next_in_list(l, 2) == 3
    assert _next_in_list(l, 3) == 3
    assert _next_in_list(l, 4) == 4


def test_best_clusters(quality):
    l = list(range(1, 5))
    assert _best_clusters(l, quality) == [4, 3, 2, 1]
    assert _best_clusters(l, quality, n_max=0) == [4, 3, 2, 1]
    assert _best_clusters(l, quality, n_max=1) == [4]
    assert _best_clusters(l, quality, n_max=2) == [4, 3]
    assert _best_clusters(l, quality, n_max=10) == [4, 3, 2, 1]


def test_most_similar_clusters(cluster_ids, similarity, status):

    def _similar(cluster):
        return _most_similar_clusters(cluster,
                                      cluster_ids=cluster_ids,
                                      similarity=similarity,
                                      status=status)

    assert not _similar(None)
    assert not _similar(100)

    assert _similar(0) == [30, 20, 2, 11, 1]
    assert _similar(1) == [30, 20, 2, 11]
    assert _similar(2) == [30, 20, 11, 1]

    assert _similar(10) == [30, 20, 2, 11, 1]
    assert _similar(11) == [30, 20, 2, 1]
    assert _similar(20) == [30, 2, 11, 1]
    assert _similar(30) == [20, 2, 11, 1]


#------------------------------------------------------------------------------
# Test strategy functions
#------------------------------------------------------------------------------

def test_best_quality_strategy(cluster_ids, quality, status, similarity):

    def _next(selection):
        return _best_quality_strategy(selection,
                                      cluster_ids=cluster_ids,
                                      quality=quality,
                                      status=status,
                                      similarity=similarity)

    assert not _next(None)
    assert _next([]) == [30]
    assert _next([30]) == [20]
    assert _next([20]) == [2]
    assert _next([2]) == [11]

    assert _next([30, 20]) == [30, 2]
    assert _next([10, 2]) == [10, 1]
    assert _next([10, 1]) == [10, 1]  # 0 is ignored, so it does not appear.


def test_best_similarity_strategy(cluster_ids, quality, status, similarity):

    def _next(selection):
        return _best_similarity_strategy(selection,
                                         cluster_ids=cluster_ids,
                                         quality=quality,
                                         status=status,
                                         similarity=similarity)

    assert not _next(None)
    assert _next([]) == [30, 20]
    assert _next([30, 20]) == [30, 11]
    assert _next([30, 11]) == [30, 2]
    assert _next([20, 10]) == [20, 2]
    assert _next([10, 2]) == [2, 1]
    assert _next([2, 1]) == [2, 1]  # 0 is ignored, so it does not appear.


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_wizard_empty():
    wizard = Wizard()
    with raises(RuntimeError):
        wizard.restart()

    wizard = Wizard()
    wizard.set_cluster_ids_function(lambda: [])
    wizard.restart()


def test_wizard_nav(wizard):
    w = wizard
    assert w.cluster_ids == [0, 1, 2, 10, 11, 20, 30]
    assert w.n_clusters == 7

    assert w.selection == []

    ###
    w.select([])
    assert w.selection == []

    assert w.best is None
    assert w.match is None

    ###
    w.select([1])
    assert w.selection == [1]

    assert w.best == 1
    assert w.match is None

    ###
    w.select([1, 2, 4])
    assert w.selection == [1, 2]

    assert w.best == 1
    assert w.match == 2

    ###
    w.previous()
    assert w.selection == [1]

    for _ in range(2):
        w.previous()
        assert w.selection == [1]

    ###
    w.next()
    assert w.selection == [1, 2]

    for _ in range(2):
        w.next()
        assert w.selection == [1, 2]


def test_wizard_next(wizard, status):
    w = wizard

    assert w.next_selection([30]) == [20]
    assert w.next_selection([30], ignore_group=True) == [20]

    # After the last good, the best ignored.
    assert w.next_selection([1]) == [10]
    # After the last unsorted (1's group is ignored), the best good.
    assert w.next_selection([1], ignore_group=True) == [11]

    @w.set_status_function
    def status_bis(cluster):
        if cluster == 30:
            return 'ignored'
        return status(cluster)

    assert w.next_selection([30]) == [10]
    assert w.next_selection([30], ignore_group=True) == [20]


def test_wizard_next_bis(wizard):
    w = wizard

    # 30, 20, 11, 10, 2, 1, 0
    #  N,  i,  g,  g, N, g, i

    @w.set_status_function
    def status_bis(cluster):
        return {0: 'ignored',
                1: 'good',
                2: None,
                10: 'good',
                11: 'good',
                20: 'ignored',
                30: None,
                }[cluster]

    wizard.select([30])
    assert wizard.next_by_quality() == [2]
    assert wizard.next_by_quality() == [11]


def test_wizard_pin_by_quality(wizard):
    w = wizard

    w.pin()
    assert w.selection == []

    w.unpin()
    assert w.selection == []

    w.next_by_quality()
    assert w.selection == [30]

    w.next_by_quality()
    assert w.selection == [20]

    # Pin.
    w.pin()
    assert w.selection == [20, 30]

    w.next_by_quality()
    assert w.selection == [20, 2]

    # Unpin.
    w.unpin()
    assert w.selection == [20]

    w.next_by_quality()
    assert w.selection == [2]

    # Pin.
    w.pin()
    assert w.selection == [2, 30]

    w.next_by_quality()
    assert w.selection == [2, 20]

    # Candidate is best among good.
    w.next_by_quality()
    assert w.selection == [2, 11]

    # Candidate is last among good, ignored are completely ignored.
    w.next_by_quality()
    assert w.selection == [2, 1]

    w.next_by_quality()
    assert w.selection == [2, 1]


def test_wizard_pin_by_similarity(wizard):
    w = wizard

    w.pin()
    assert w.selection == []

    w.unpin()
    assert w.selection == []

    w.next_by_similarity()
    assert w.selection == [30, 20]

    w.next_by_similarity()
    assert w.selection == [30, 11]

    w.pin()
    assert w.selection == [30, 20]

    w.next_by_similarity()
    assert w.selection == [30, 11]

    w.unpin()
    assert w.selection == [30]

    w.select([20, 10])
    assert w.selection == [20, 10]

    w.next_by_similarity()
    assert w.selection == [20, 2]

    w.next_by_similarity()
    assert w.selection == [20, 1]

    w.next_by_similarity()
    assert w.selection == [11, 2]
