# -*- coding: utf-8 -*-

"""Tests of manual clustering utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ....utils.logging import set_level, debug
from .._utils import ClusterMetadataUpdater, UpdateInfo
from ....io.kwik.model import ClusterMetadata


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def teardown():
    set_level('info')


def test_metadata_history():
    """Test ClusterMetadataUpdater history."""

    data = {2: {'group': 2, 'color': 7}, 4: {'group': 5}}

    base_meta = ClusterMetadata(data=data)

    @base_meta.default
    def group(cluster):
        return 3

    @base_meta.default
    def color(cluster):
        return 0

    meta = ClusterMetadataUpdater(base_meta)

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


def test_update_info():
    debug(UpdateInfo(deleted=range(5), added=[5], description='merge'))
    debug(UpdateInfo(deleted=range(5), added=[5], description='assign'))
    debug(UpdateInfo(deleted=range(5), added=[5],
                     description='assign', history='undo'))
