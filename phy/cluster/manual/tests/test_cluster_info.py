# -*- coding: utf-8 -*-

"""Tests of cluster metadata."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ....ext.six import itervalues, iterkeys
from ..cluster_info import (ClusterMetadata,
                            ClusterStats)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_cluster_metadata():
    meta = ClusterMetadata()

    @meta.default
    def group(cluster):
        return 3

    @meta.default
    def color(cluster):
        return 0

    assert meta.group(0) is not None
    assert meta.group(2) == 3
    assert meta.group(10) == 3

    meta.set_color(10, 5)
    assert meta.color(10) == 5

    # Alternative __setitem__ syntax.
    info = meta.set_color([10, 11], 5)
    assert meta.color(10) == 5
    assert meta.color(11) == 5
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [10, 11]

    info = meta.set_color([10, 11], 6)
    assert meta.color(10) == 6
    assert meta.color(11) == 6
    assert meta.color([10, 11]) == [6, 6]
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [10, 11]

    meta.set_color(10, 20)
    assert meta.color(10) == 20


def test_metadata_history():
    """Test ClusterMetadata history."""

    data = {2: {'group': 2, 'color': 7}, 4: {'group': 5}}

    meta = ClusterMetadata(data=data)

    @meta.default
    def group(cluster):
        return 3

    @meta.default
    def color(cluster):
        return 0

    # Values set in 'data'.
    assert meta.group(2) == 2
    assert meta.color(2) == 7

    # Default values.
    assert meta.group(3) == 3
    assert meta.color(3) != 7

    assert meta.group(4) == 5
    assert meta.color(4) != 7

    ###########

    meta.undo()
    meta.redo()

    # Action 1.
    info = meta.set_group(2, 20)
    assert meta.group(2) == 20
    assert info.description == 'metadata_group'
    assert info.metadata_changed == [2]

    # Action 2.
    info = meta.set_color(3, 30)
    assert meta.color(3) == 30
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [3]

    # Action 3.
    info = meta.set_color(2, 40)
    assert meta.color(2) == 40
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [2]

    ###########

    # Undo 3.
    info = meta.undo()
    assert meta.color(2) == 7
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [2]

    # Undo 2.
    info = meta.undo()
    assert meta.color(3) != 7
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [3]

    # Redo 2.
    info = meta.redo()
    assert meta.color(3) == 30
    assert meta.group(2) == 20
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [3]

    # Undo 2.
    info = meta.undo()
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [3]

    # Undo 1.
    info = meta.undo()
    assert meta.group(2) == 2
    assert info.description == 'metadata_group'
    assert info.metadata_changed == [2]

    info = meta.undo()
    assert info is None

    info = meta.undo()
    assert info is None


def test_stats():

    stats = ClusterStats()

    class O(object):
        coeff = 2

        def my_stat(self, x):
            return self.coeff * x

    o = O()

    # Register the statistics.
    stats.stat(o.my_stat)

    assert stats.my_stat(3) == 6

    o.coeff = 3
    assert stats.my_stat(3) == 6

    stats.invalidate(3)
    assert stats.my_stat(3) == 9
