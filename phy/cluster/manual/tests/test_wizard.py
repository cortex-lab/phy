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

    wizard = Wizard()
    wizard.cluster_ids = [2, 3, 5]

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
    assert wizard.best_cluster() == 2
    assert wizard.most_similar_clusters() == [5, 3]
    assert wizard.most_similar_clusters(2) == [5, 3]
    wizard.mark_dissimilar(2, 3)
