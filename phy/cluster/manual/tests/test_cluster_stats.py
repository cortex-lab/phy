# -*- coding: utf-8 -*-

"""Tests of cluster stats."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ....ext.six import itervalues, iterkeys
from ..cluster_stats import ClusterStats


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_stats():

    stats = ClusterStats()

    class O(object):
        coeff = 2

        def my_stat(self, x):
            return self.coeff * x

    o = O()

    stats = ClusterStats(my_stat=o.my_stat)
    assert stats.get(3, 'my_stat') == 6
    assert stats.my_stat(3) == 6

    o.coeff = 3
    assert stats.get(3, 'my_stat') == 6
    assert stats.my_stat(3) == 6

    stats.invalidate(3)
    assert stats.get(3, 'my_stat') == 9
    assert stats.my_stat(3) == 9
