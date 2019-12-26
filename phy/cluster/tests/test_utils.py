# -*- coding: utf-8 -*-

"""Tests of manual clustering utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from pytest import raises

from .._utils import (ClusterMeta, UpdateInfo, RotatingProperty,
                      _update_cluster_selection, create_cluster_meta)

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_create_cluster_meta():
    cluster_groups = {2: 3,
                      3: 3,
                      5: 1,
                      7: 2,
                      }
    meta = create_cluster_meta(cluster_groups)
    assert meta.group(2) == 3
    assert meta.group(3) == 3
    assert meta.group(5) == 1
    assert meta.group(7) == 2
    assert meta.group(8) is None


def test_metadata_history_simple():
    """Test ClusterMeta history."""

    meta = ClusterMeta()

    # The 'group' field is automatically created.
    meta.set('group', 2, 2)
    assert meta.get('group', 2) == 2

    meta.undo()
    assert meta.get('group', 2) is None

    meta.redo()
    assert meta.get('group', 2) == 2

    with raises(AssertionError):
        assert meta.to_dict('grou') is None
    assert meta.to_dict('group') == {2: 2}


def test_metadata_history_complex():
    """Test ClusterMeta history."""

    meta = ClusterMeta()
    meta.add_field('group', 3)
    meta.add_field('color', 0)

    data = {2: {'group': 2, 'color': 7}, 4: {'group': 5}}
    meta.from_dict(data)

    assert meta.group(2) == 2
    assert meta.group([4, 2]) == [5, 2]

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
    info = meta.set('group', 2, 20)
    assert meta.group(2) == 20
    assert info.description == 'metadata_group'
    assert info.metadata_changed == [2]

    # Action 2.
    info = meta.set('color', 3, 30)
    assert meta.color(3) == 30
    assert info.description == 'metadata_color'
    assert info.metadata_changed == [3]

    # Action 3.
    info = meta.set('color', 2, 40)
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


def test_metadata_descendants():
    """Test ClusterMeta history."""

    data = {0: {'group': 0},
            1: {'group': 1},
            2: {'group': 2},
            3: {'group': 3},
            }

    meta = ClusterMeta()

    meta.add_field('group', 3)
    meta.from_dict(data)

    meta.set_from_descendants([])
    assert meta.group(4) == 3

    meta.set_from_descendants([(0, 4)])
    assert meta.group(4) == 0

    # Reset to default.
    meta.set('group', 4, 3)
    meta.set_from_descendants([(1, 4)])
    assert meta.group(4) == 1

    meta.set_from_descendants([(1, 5), (2, 5)], largest_old_cluster=2)
    # This is the value of the largest old cluster.
    assert meta.group(5) == 2

    # The old clusters 2 and 3 have the same value, so we set it to the new cluster.
    meta.set('group', 3, 2)
    meta.set_from_descendants([(2, 6), (3, 6), (10, 10)])
    assert meta.group(6) == 2

    # If the value of the new cluster is non-default, it should not
    # be changed by set_from_descendants.
    meta.set_from_descendants([(0, 2)])
    assert meta.group(2) == 2


def test_update_cluster_selection():
    clusters = [1, 2, 3]
    up = UpdateInfo(deleted=[2], added=[4, 0])
    assert _update_cluster_selection(clusters, up) == [1, 3, 4, 0]


def test_update_info():
    logger.debug(UpdateInfo())
    logger.debug(UpdateInfo(description='hello'))
    logger.debug(UpdateInfo(deleted=range(5), added=[5], description='merge'))
    logger.debug(UpdateInfo(deleted=range(5), added=[5], description='assign'))
    logger.debug(UpdateInfo(deleted=range(5), added=[5],
                            description='assign', history='undo'))
    logger.debug(UpdateInfo(metadata_changed=[2, 3], description='metadata'))


def test_rotating_property():
    rp = RotatingProperty()
    rp.add('f1', 1)
    rp.add('f2', 2)
    rp.add('f3', 3)

    assert rp.current == 'f1'
    rp.next()
    assert rp.current == 'f2'

    rp.set('f3')
    assert rp.get() == 3
    assert rp.next() == 'f1'
    assert rp.previous() == 'f3'
