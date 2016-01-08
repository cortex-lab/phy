# -*- coding: utf-8 -*-

"""Test cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..store import ClusterStore
from phy.io import Context


#------------------------------------------------------------------------------
# Test cluster stats
#------------------------------------------------------------------------------

def test_cluster_store(tempdir):
    context = Context(tempdir)
    cs = ClusterStore(context=context)

    @cs.add(cache='memory')
    def f(x):
        return x * x

    assert cs.f(3) == 9
    assert cs.get('f')(3) == 9
