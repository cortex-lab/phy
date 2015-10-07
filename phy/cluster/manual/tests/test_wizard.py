# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from numpy.testing import assert_array_equal as ae

from ..wizard import (_argsort,
                      _best_clusters,
                      Wizard,
                      _wizard_group,
                      )


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


def test_best_clusters():
    quality = lambda c: c * .1
    l = list(range(1, 5))
    assert _best_clusters(l, quality) == [4, 3, 2, 1]
    assert _best_clusters(l, quality, n_max=0) == [4, 3, 2, 1]
    assert _best_clusters(l, quality, n_max=1) == [4]
    assert _best_clusters(l, quality, n_max=2) == [4, 3]
    assert _best_clusters(l, quality, n_max=10) == [4, 3, 2, 1]


def test_wizard_group():
    assert _wizard_group('noise') == 'ignored'
    assert _wizard_group('mua') == 'ignored'
    assert _wizard_group('good') == 'good'
    assert _wizard_group('unknown') is None
    assert _wizard_group(None) is None


def test_wizard_core(mock_wizard):

    w = mock_wizard

    assert w.best_clusters() == [3, 2, 1]
    assert w.best_clusters(n_max=0) == [3, 2, 1]
    assert w.best_clusters(n_max=None) == [3, 2, 1]
    assert w.best_clusters(n_max=2) == [3, 2]
    assert w.best_clusters(n_max=1) == [3]

    assert w.most_similar_clusters(3) == [2, 1]
    assert w.most_similar_clusters(2) == [3, 1]
    assert w.most_similar_clusters(1) == [3, 2]

    assert w.most_similar_clusters(3, n_max=0) == [2, 1]
    assert w.most_similar_clusters(3, n_max=None) == [2, 1]
    assert w.most_similar_clusters(3, n_max=1) == [2]
    assert w.most_similar_clusters(3, n_max=2) == [2, 1]
    assert w.most_similar_clusters(3, n_max=10) == [2, 1]
