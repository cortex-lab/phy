# -*- coding: utf-8 -*-

"""Tests of cluster metadata."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ....ext.six import itervalues, iterkeys
from ..cluster_metadata import (_cluster_info, ClusterMetadata,
                                ClusterDefaultDict, ClusterStats,
                                _fun_arg_count)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_fun_arg_count():

    def f():
        pass

    assert _fun_arg_count(f) == 0

    def f(x):
        pass

    assert _fun_arg_count(f) == 1

    def f(x, y=0):
        pass

    assert _fun_arg_count(f) == 2

    class O(object):
        def f(self):
            pass

    assert _fun_arg_count(O.f) == 0

    class O(object):
        def f(self, x):
            pass

    assert _fun_arg_count(O.f) == 1

    class O(object):
        def f(self, x, y=None):
            pass

    assert _fun_arg_count(O.f) == 2


def test_cluster_default_dict():

    class Factory(object):
        _no_args_called = False
        _one_arg_called = None

        def no_args(self):
            self._no_args_called = True
            return 'default'

        def one_arg(self, key):
            self._one_arg_called = key
            return 'default {0}'.format(key)

    factory = Factory()

    my_dict = ClusterDefaultDict(factory.no_args)
    assert my_dict[3] == 'default'
    assert factory._no_args_called

    my_dict = ClusterDefaultDict(factory.one_arg)
    assert my_dict[7] == 'default 7'
    assert factory._one_arg_called


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
    meta = ClusterMetadata(fields=[('field', lambda: 9)])
    assert meta.data[3]['field'] == 9


def test_cluster_metadata():
    meta = ClusterMetadata()
    meta.update()
    assert meta.data is not None

    assert meta[0]['group'] is not None

    # assert meta[2]['color'] == 1
    assert meta[2]['group'] == 3

    # assert meta[10]['color'] == 1
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
    assert meta.get(3, 'color') != 7

    assert meta.get(4, 'group') == 5
    assert meta.get(4, 'color') != 7

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
    assert meta.get(3, 'color') != 7
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
