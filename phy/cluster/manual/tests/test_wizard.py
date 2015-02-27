# -*- coding: utf-8 -*-

"""Test wizard."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import numpy.random as npr
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ..wizard import Wizard
from ..cluster_info import ClusterMetadata


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_wizard():

    wizard = Wizard()
    wizard.cluster_ids = [2, 3, 5]

    @wizard.quality
    def quality(cluster):
        return {2: .9,
                3: .3,
                5: .6,
                }[cluster]

    @wizard.similarity
    def similarity(cluster, other):
        cluster, other = min((cluster, other)), max((cluster, other))
        return {(2, 3): 1,
                (2, 5): 2,
                (3, 5): 3}[cluster, other]

    assert wizard.best_clusters() == [2, 5, 3]
    assert wizard.best_cluster() == 2
    assert wizard.most_similar_clusters(2) == [5, 3]
    wizard.mark_dissimilar(2, 3)
