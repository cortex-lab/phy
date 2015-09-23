# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from ..clustering import Clustering
from .._utils import ClusterMetadata, ClusterMetadataUpdater
from ..wizard import (_previous,
                      _next,
                      Wizard,
                      )


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def wizard():

    def get_cluster_ids():
        return [2, 3, 5, 7]

    wizard = Wizard()
    wizard.set_cluster_ids_function(get_cluster_ids)

    @wizard.set_status_function
    def cluster_status(cluster):
        return {2: None, 3: None, 5: 'ignored', 7: 'good'}.get(cluster, None)

    @wizard.set_quality_function
    def quality(cluster):
        return cluster * .1

    @wizard.set_similarity_function
    def similarity(cluster, other):
        return 1. + quality(cluster) - quality(other)

    yield wizard


@yield_fixture
def cluster_metadata():
    data = {2: {'group': 3},
            3: {'group': 3},
            5: {'group': 1},
            7: {'group': 2},
            }

    base_meta = ClusterMetadata(data=data)

    @base_meta.default
    def group(cluster):
        return 3

    meta = ClusterMetadataUpdater(base_meta)
    yield meta


@yield_fixture
def clustering():
    yield Clustering([2, 3, 5, 7])


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_utils():
    l = [2, 3, 5, 7, 11]

    def func(x):
        return x in (2, 5)

    # Error: log and do nothing.
    _previous(l, 1, func)
    _previous(l, 15, func)

    assert _previous(l, 2, func) == 2
    assert _previous(l, 3, func) == 2
    assert _previous(l, 5, func) == 2
    assert _previous(l, 7, func) == 5
    assert _previous(l, 11, func) == 5

    # Error: log and do nothing.
    _next(l, 1, func)
    _next(l, 15, func)

    assert _next(l, 2, func) == 5
    assert _next(l, 3, func) == 5
    assert _next(l, 5, func) == 5
    assert _next(l, 7, func) == 7
    assert _next(l, 11, func) == 11


def test_wizard_core():

    wizard = Wizard()
    wizard.set_cluster_ids_function(lambda: [2, 3, 5])

    @wizard.set_quality_function
    def quality(cluster):
        return {2: .9,
                3: .3,
                5: .6,
                }[cluster]

    @wizard.set_similarity_function
    def similarity(cluster, other):
        cluster, other = min((cluster, other)), max((cluster, other))
        return {(2, 3): 1,
                (2, 5): 2,
                (3, 5): 3}[cluster, other]

    assert wizard.best_clusters() == [2, 5, 3]
    assert wizard.best_clusters(n_max=0) == [2, 5, 3]
    assert wizard.best_clusters(n_max=None) == [2, 5, 3]
    assert wizard.best_clusters(n_max=2) == [2, 5]

    assert wizard.best_clusters(n_max=1) == [2]

    assert wizard.most_similar_clusters() == [5, 3]
    assert wizard.most_similar_clusters(2) == [5, 3]

    assert wizard.most_similar_clusters(n_max=0) == [5, 3]
    assert wizard.most_similar_clusters(n_max=None) == [5, 3]
    assert wizard.most_similar_clusters(n_max=1) == [5]


def test_wizard_nav(wizard):

    # Loop over the best clusters.
    wizard.start()

    assert wizard.n_clusters == 4
    assert wizard.best_list == [3, 2, 7, 5]

    assert wizard.best == 3
    assert wizard.match is None

    wizard.next()
    assert wizard.best == 2

    wizard.previous()
    assert wizard.best == 3

    wizard.previous_best()
    assert wizard.best == 3

    wizard.next()
    assert wizard.best == 2

    wizard.next()
    assert wizard.best == 7

    wizard.next_best()
    assert wizard.best == 5

    wizard.next()
    assert wizard.best == 5

    # Now we start again.
    wizard.start()
    assert wizard.best == 3
    assert wizard.match is None

    # The match are sorted by group first (unsorted, good, and ignored),
    # and similarity second.
    wizard.pin()
    assert wizard.best == 3
    assert wizard.match == 2
    assert wizard.match_list == [2, 7, 5]

    wizard.next()
    assert wizard.best == 3
    assert wizard.match == 7

    wizard.next_match()
    assert wizard.best == 3
    assert wizard.match == 5

    wizard.previous_match()
    assert wizard.best == 3
    assert wizard.match == 7

    wizard.previous()
    assert wizard.best == 3
    assert wizard.match == 2

    wizard.first()
    assert wizard.selection == [3, 2]
    wizard.last()
    assert wizard.selection == [3, 5]

    wizard.unpin()
    assert wizard.best == 3
    assert wizard.match is None

    assert wizard.n_processed == 2


def test_wizard_update(wizard, clustering, cluster_metadata):
    # 2: none, 3: none, 5: unknown, 7: good
    wizard.attach(clustering, cluster_metadata)

    wizard.start()

    def _check_best_match(b, m):
        assert wizard.selection == [b, m]
        assert wizard.best == b
        assert wizard.match == m

    assert wizard.best_list == [3, 2, 7, 5]
    wizard.next()
    wizard.pin()

    _check_best_match(2, 3)

    # Save the selection before the merge in the undo stack.
    clustering.merge([2, 3])
    assert wizard.best_list == [8, 7, 5]
    _check_best_match(8, 7)

    # Undo.
    clustering.undo()
    _check_best_match(2, 3)

    wizard.selection = [1, 5, 7, 8]
    _check_best_match(5, 7)

    # Redo.
    clustering.redo()
    # _check_best_match(8, 7)
