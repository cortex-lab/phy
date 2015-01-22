# -*- coding: utf-8 -*-

"""Tests of sparse matrix structures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from numpy.testing import assert_array_equal
from pytest import raises

from ....ext.six import itervalues, iterkeys
from ..cluster_metadata import _cluster_info, ClusterMetadata


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_structure():
    """Test the structure holding all cluster metadata."""
    data = _cluster_info([('a', 1), ('b', 2)])

    assert isinstance(data[3], dict)
    assert data[3]['a'] == 1
    assert data[3]['b'] == 2

    data[3]['b'] = 10
    assert data[3]['b'] == 10

    with raises(KeyError):
        data[3]['c']


def test_default_function():
    meta = ClusterMetadata([('field', lambda: 9)])
    assert meta.data[3]['field'] == 9


def test_cluster_metadata():
    meta = ClusterMetadata()
    assert meta.data is not None

    assert meta[0]['group'] is not None

    assert meta[2]['color'] == 1
    assert meta[2]['group'] == 3

    assert meta[10]['color'] == 1
    assert meta[10]['group'] == 3

    info = meta.set([10], 'color', 5)
    assert meta[10]['color'] == 5
    assert info.description == 'color'
    assert info.metadata_changed == [10]

    # Alternative __setitem__ syntax.
    info = meta.set([10, 11], 'color', 5)
    assert meta[10]['color'] == 5
    assert meta[11]['color'] == 5
    assert info.description == 'color'
    assert info.metadata_changed == [10, 11]

    info = meta.set([10, 11], 'color', [6, 7])
    assert meta[10]['color'] == 6
    assert meta.data[11]['color'] == 7
    assert info.description == 'color'
    assert info.metadata_changed == [10, 11]

    meta[10]['color'] = 10
    assert meta[10]['color'] == 10


def test_metadata_history():
    """Test ClusterMetadata history."""

    data = {2: {'group': 2, 'color': 7}, 4: {'group': 5}}

    meta = ClusterMetadata(data=data)

    # Values set in 'data'.
    assert meta.get(2, 'group') == 2
    assert meta.get(2, 'color') == 7

    # Default values.
    assert meta.get(3, 'group') == 3
    assert meta.get(3, 'color') == 1

    assert meta.get(4, 'group') == 5
    assert meta.get(4, 'color') == 1

    ###########

    meta.undo()
    meta.redo()

    # Action 1.
    info = meta.set(2, 'group', 20)
    assert meta.get(2, 'group') == 20
    assert info.description == 'group'
    assert info.metadata_changed == [2]

    # Action 2.
    info = meta.set(3, 'color', 30)
    assert meta.get(3, 'color') == 30
    assert info.description == 'color'
    assert info.metadata_changed == [3]

    # Action 3.
    info = meta.set(2, 'color', 40)
    assert meta.get(2, 'color') == 40
    assert info.description == 'color'
    assert info.metadata_changed == [2]

    ###########

    # Undo 3.
    info = meta.undo()
    assert meta.get(2, 'color') == 7
    assert info.description == 'color'
    assert info.metadata_changed == [2]

    # Undo 2.
    info = meta.undo()
    assert meta.get(3, 'color') == 1
    assert info.description == 'color'
    assert info.metadata_changed == [3]

    # Redo 2.
    info = meta.redo()
    assert meta.get(3, 'color') == 30
    assert meta.get(2, 'group') == 20
    assert info.description == 'color'
    assert info.metadata_changed == [3]

    # Undo 2.
    info = meta.undo()
    assert info.description == 'color'
    assert info.metadata_changed == [3]

    # Undo 1.
    info = meta.undo()
    assert meta.get(2, 'group') == 2
    assert info.description == 'group'
    assert info.metadata_changed == [2]

    info = meta.undo()
    assert info is None

    info = meta.undo()
    assert info is None
