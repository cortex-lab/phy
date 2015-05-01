# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..wizard import Wizard


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_core_wizard():

    wizard = Wizard([2, 3, 5])

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

    assert wizard.best_cluster() == 2

    assert wizard.most_similar_clusters() == [5, 3]
    assert wizard.most_similar_clusters(2) == [5, 3]

    assert wizard.most_similar_clusters(n_max=0) == [5, 3]
    assert wizard.most_similar_clusters(n_max=None) == [5, 3]
    assert wizard.most_similar_clusters(n_max=1) == [5]

    # Test ignore cluster.
    assert wizard.best_clusters() == [2, 5, 3]
    wizard.ignore(2)
    assert wizard.best_clusters() == [5, 3]

    # Test ignore pair of clusters.
    assert wizard.most_similar_clusters(2) == [5, 3]
    wizard.ignore((2, 5))
    assert wizard.most_similar_clusters(2) == [3]


def test_pin():

    n = 20
    clusters = list(range(n))
    wizard = Wizard(clusters)

    @wizard.set_quality_function
    def quality(cluster):
        return cluster / float(n)

    @wizard.set_similarity_function
    def similarity(cluster, other):
        d = max(0, other - cluster)
        if d == 0:
            return 0.
        else:
            return 1. - (d - 1) / float(n)

    assert not wizard.is_running()
    assert wizard.count() == 0

    wizard.start()
    assert wizard.is_running()

    # Test moves.
    assert wizard.count() == n
    assert wizard.index() == 0

    wizard.next()
    assert wizard.index() == 1

    wizard.last()
    assert wizard.index() == n - 1

    wizard.first()

    # Test pin.
    wizard.next()
    wizard.next()
    assert wizard.current_selection() == (n - 3,)
    wizard.pin()

    # Go through the closest matches.
    assert wizard.current_selection() == (n - 3, n - 2)
    assert wizard.next() == n - 1
    assert wizard.current_selection() == (n - 3, n - 1)
    wizard.next()

    # Test playback methods.
    wizard.first()
    wizard.next()
    wizard.previous()
    assert wizard.index() == 0

    wizard.previous()
    assert wizard.index() == 0

    wizard.pause()
    assert not wizard.is_running()

    wizard.start()
    assert wizard.is_running()

    wizard.stop()
    assert not wizard.is_running()
