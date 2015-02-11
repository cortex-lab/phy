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


#------------------------------------------------------------------------------
# Test wizard
#------------------------------------------------------------------------------

def test_wizard():

    n_channels = 8

    def _masks(*ind_true):
        m = np.zeros(n_channels, dtype=np.bool)
        m[np.array(ind_true)] = True
        return m

    # 2-3: 1
    # 2-5: 2
    # 3-5: 3
    cluster_stats = {2: {'quality': .9,
                         'cluster_masks': _masks(1, 3)},
                     3: {'quality': .3,
                         'cluster_masks': _masks(2, 3, 4)},
                     5: {'quality': .6,
                         'cluster_masks': _masks(1, 2, 3, 4, 5)}}

    wizard = Wizard(cluster_stats=cluster_stats,
                    cluster_metadata={})

    assert wizard.best_clusters() == [2, 5, 3]
    assert wizard.best_cluster() == 2
    assert wizard.most_similar_clusters(2) == [5, 3]
    wizard.mark_dissimilar(2, 3)
