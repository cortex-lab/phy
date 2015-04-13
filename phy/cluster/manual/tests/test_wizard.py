# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..wizard import Wizard


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_wizard():

    wizard = Wizard([2, 3, 5])

    @wizard.set_quality
    def quality(cluster):
        return {2: .9,
                3: .3,
                5: .6,
                }[cluster]

    @wizard.set_similarity
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
